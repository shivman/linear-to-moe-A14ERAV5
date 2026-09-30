"""
ERA V5 Session 14 — train a dense linear model, then turn it into an MoE.

The dense model is a small GPT. Its feed-forward block is one linear map
(GELU MLP). After that loss is falling, each block is upcycled:

  copy — every expert starts as that same MLP; only the router is new
  drop — same copy, then half the hidden units are redrawn (r = 0.5)
  partition — the MLP becomes a shared expert (always on); each routed
              expert keeps a random overlapping half of its units

Both keep training on the same TinyStories stream. A third arm keeps the
dense MLP and trains the same number of extra steps, so the loss after the
split is not only an MoE story.

Router (lecture §7): softmax over experts, keep top-k, rescale so the kept
weights sum to 1. Router math is fp32. A per-expert bias is added only when
choosing, then nudged by sign(mean load − load) at 0.001. No token dropping.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
FIGS = ROOT / "figures"
A13_TEXT = (
    ROOT.parents[1]
    / "Assignment_13"
    / "session13_reversible"
    / "data"
    / "tinystories_12000000.txt"
)
A13_TOK = (
    ROOT.parents[1]
    / "Assignment_13"
    / "session13_reversible"
    / "data"
    / "tokenizer.json"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


@dataclass
class Cfg:
    d: int = 128
    n_layer: int = 4
    n_head: int = 4
    vocab: int = 4096
    T: int = 64
    inner: int = 512
    experts: int = 8
    k: int = 2
    lr: float = 3e-3
    weight_decay: float = 0.01
    bias_step: float = 0.001
    drop_ratio: float = 0.5
    seed: int = 14


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def router_example() -> dict:
    """Lecture token: logits for 8 experts, softmax, top-2 rescaled."""
    logits = torch.tensor([2.0, 0.5, 1.2, -0.3, 0.9, 1.8, -1.0, 0.1])
    scores = torch.softmax(logits, dim=0)
    keep = torch.topk(scores, 2)
    weights = keep.values / keep.values.sum()
    rec = {
        "softmax_0": round(float(scores[0]), 3),
        "softmax_5": round(float(scores[5]), 3),
        "weight_0": round(float(weights[0]), 3),
        "weight_5": round(float(weights[1]), 3),
        "kept_experts": [int(i) for i in keep.indices],
    }
    assert rec["kept_experts"] == [0, 5], rec
    assert abs(rec["softmax_0"] - 0.320) < 0.002
    assert abs(rec["weight_0"] - 0.550) < 0.002
    assert abs(rec["weight_0"] + rec["weight_5"] - 1.0) < 1e-6
    print(
        f"router check: softmax[{rec['softmax_0']}, {rec['softmax_5']}] "
        f"→ weights [{rec['weight_0']}, {rec['weight_5']}]"
    )
    return rec


class MLP(nn.Module):
    """One dense feed-forward block: the linear model the assignment starts from."""

    def __init__(self, d: int, inner: int):
        super().__init__()
        self.fc1 = nn.Linear(d, inner)
        self.fc2 = nn.Linear(inner, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoE(nn.Module):
    """Dropless top-k. Bias changes who is chosen, not the output weights."""

    def __init__(
        self,
        d: int,
        inner: int,
        experts: int,
        k: int,
        bias_step: float,
        shared_inner: int | None = None,
    ):
        super().__init__()
        self.k = k
        self.bias_step = bias_step
        self.router = nn.Linear(d, experts, bias=False)
        nn.init.normal_(self.router.weight, mean=0.0, std=0.002)
        self.experts = nn.ModuleList([MLP(d, inner) for _ in range(experts)])
        self.shared = MLP(d, shared_inner) if shared_inner else None
        self.route = "hard"
        self.register_buffer("bias", torch.zeros(experts))
        self.last_load: torch.Tensor | None = None
        self._token_scores: torch.Tensor | None = None

    def _choose(self, scores: torch.Tensor) -> torch.Tensor:
        choose = scores + self.bias.float()
        if self.route == "prob":
            # Gumbel-top-k: each expert is drawn with probability proportional to its score.
            u = torch.rand_like(choose).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            pick = torch.log(choose.clamp_min(1e-8)) + gumbel
            return torch.topk(pick, self.k, dim=-1).indices
        return torch.topk(choose, self.k, dim=-1).indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = x.reshape(-1, d)
        logits = F.linear(h.float(), self.router.weight.float())
        scores = torch.softmax(logits, dim=-1)
        self._token_scores = scores
        topi = self._choose(scores)
        chosen = scores.gather(-1, topi)
        weights = chosen / chosen.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(h)
        n = h.shape[0]
        for e, expert in enumerate(self.experts):
            slot = topi == e
            if not bool(slot.any()):
                continue
            picked = slot.any(dim=-1)
            ye = expert(h[picked])
            we = (weights * slot.float()).sum(dim=-1)[picked]
            out[picked] = out[picked] + ye * we.unsqueeze(-1)
        if self.shared is not None:
            out = out + self.shared(h)
        counts = torch.bincount(topi.reshape(-1), minlength=self.bias.numel()).float()
        self.last_load = counts / float(n)
        return out.view(b, t, d).to(dtype=x.dtype)

    @torch.no_grad()
    def update_bias(self) -> None:
        if self.last_load is None:
            return
        load = self.last_load
        self.bias += self.bias_step * torch.sign(load.mean() - load)


class Block(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        d, h = cfg.d, cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.mlp: nn.Module = MLP(d, cfg.inner)
        self.n_head = h
        self.d_head = d // h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, t, 3, self.n_head, self.d_head)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = torch.softmax(att.masked_fill(mask, torch.finfo(att.dtype).min), dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d)
        self.pos = nn.Embedding(cfg.T, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        scale = 0.02 / math.sqrt(2.0 * self.cfg.n_layer)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, mean=0.0, std=scale)
            nn.init.normal_(blk.mlp.fc2.weight, mean=0.0, std=scale)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def moe_modules(self) -> list[MoE]:
        return [blk.mlp for blk in self.blocks if isinstance(blk.mlp, MoE)]


def redraw_half(mlp: MLP, ratio: float, n_layer: int, seed: int) -> int:
    inner = mlp.fc1.out_features
    n_drop = int(inner * ratio)
    g = torch.Generator()
    g.manual_seed(seed)
    drop = torch.randperm(inner, generator=g)[:n_drop]
    std_out = 0.02 / math.sqrt(2.0 * n_layer)
    with torch.no_grad():
        device = mlp.fc1.weight.device
        mlp.fc1.weight[drop] = torch.randn(n_drop, mlp.fc1.in_features, generator=g).to(device) * 0.02
        mlp.fc1.bias[drop] = 0
        mlp.fc2.weight[:, drop] = torch.randn(mlp.fc2.out_features, n_drop, generator=g).to(device) * std_out
    return n_drop


def half_indices(inner: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(inner, generator=g)[: inner // 2]


def take_slice(dense: MLP, idx: torch.Tensor) -> MLP:
    """One expert owns a subset of the dense hidden units. Output bias stays on the shared expert."""
    expert = MLP(dense.fc1.in_features, int(idx.numel()))
    with torch.no_grad():
        expert.fc1.weight.copy_(dense.fc1.weight[idx])
        expert.fc1.bias.copy_(dense.fc1.bias[idx])
        expert.fc2.weight.copy_(dense.fc2.weight[:, idx])
        expert.fc2.bias.zero_()
    return expert


def _shapes(dense: MLP, moe: MoE) -> dict:
    return {
        "dense_fc1": list(dense.fc1.weight.shape),
        "dense_fc2": list(dense.fc2.weight.shape),
        "expert_fc1": list(moe.experts[0].fc1.weight.shape),
        "expert_fc2": list(moe.experts[0].fc2.weight.shape),
        "router": list(moe.router.weight.shape),
        "shared_fc1": list(moe.shared.fc1.weight.shape) if moe.shared is not None else None,
    }


def upcycle(model: GPT, method: str) -> dict:
    """Replace every dense MLP with experts. `method` is copy, drop, or partition."""
    cfg = model.cfg
    info = {"method": method, "layers": cfg.n_layer, "experts": cfg.experts, "k": cfg.k}
    dropped = []
    shape = None
    for li, blk in enumerate(model.blocks):
        dense = blk.mlp
        assert isinstance(dense, MLP)
        if method == "partition":
            moe = MoE(
                cfg.d,
                cfg.inner // 2,
                cfg.experts,
                cfg.k,
                cfg.bias_step,
                shared_inner=cfg.inner,
            )
            moe.shared.load_state_dict(dense.state_dict())
            for ei, expert in enumerate(moe.experts):
                idx = half_indices(cfg.inner, cfg.seed + 1000 * li + ei)
                expert.load_state_dict(take_slice(dense, idx).state_dict())
        else:
            moe = MoE(cfg.d, cfg.inner, cfg.experts, cfg.k, cfg.bias_step)
            for ei, expert in enumerate(moe.experts):
                expert.load_state_dict(dense.state_dict())
                if method == "drop":
                    n_drop = redraw_half(expert, cfg.drop_ratio, cfg.n_layer, seed=cfg.seed + 1000 * li + ei)
                    dropped.append(n_drop)
        shape = _shapes(dense, moe)
        blk.mlp = moe
    info["neurons_redrawn_per_expert"] = dropped[0] if dropped else 0
    info["expert_inner"] = cfg.inner // 2 if method == "partition" else cfg.inner
    info["shared_inner"] = cfg.inner if method == "partition" else 0
    info["shapes"] = shape
    info["params"] = n_params(model)
    return info


def grow_experts(model: GPT, clones: int, redraw: float) -> dict:
    """Clone every expert. `redraw=0` keeps the copies identical. `redraw=0.5` redraws half of each clone."""
    cfg = model.cfg
    families = 0
    for li, blk in enumerate(model.blocks):
        old = blk.mlp
        assert isinstance(old, MoE)
        families = len(old.experts)
        inner = old.experts[0].fc1.out_features
        device = old.router.weight.device
        moe = MoE(cfg.d, inner, families * clones, old.k, cfg.bias_step, shared_inner=None).to(device)
        with torch.no_grad():
            tiled = old.router.weight.detach().repeat(clones, 1)
            noise = 1e-5 if redraw == 0 else 0.002
            moe.router.weight.copy_(tiled + torch.randn_like(tiled) * noise)
            moe.bias.copy_(old.bias.detach().repeat(clones))
        for e, expert in enumerate(old.experts):
            for c in range(clones):
                dst = moe.experts[e * clones + c]
                dst.load_state_dict(expert.state_dict())
                if redraw > 0:
                    redraw_half(dst, redraw, cfg.n_layer, seed=cfg.seed + 17 * li + 100 * e + c)
        blk.mlp = moe
    return {
        "families": families,
        "clones": clones,
        "experts": families * clones,
        "redraw": redraw,
        "params": n_params(model),
    }


def load_ids(cfg: Cfg) -> torch.Tensor:
    from tokenizers import Tokenizer

    DATA.mkdir(exist_ok=True)
    tok_path = DATA / "tokenizer.json"
    if not tok_path.exists():
        tok_path.write_bytes(A13_TOK.read_bytes())
    text_path = DATA / "tinystories_12000000.txt"
    if not text_path.exists():
        text_path.write_bytes(A13_TEXT.read_bytes())
    tok = Tokenizer.from_file(str(tok_path))
    cfg.vocab = tok.get_vocab_size()
    cache = DATA / "ids.pt"
    if cache.exists():
        ids = torch.load(cache, weights_only=True)
    else:
        ids = torch.tensor(tok.encode(text_path.read_text(encoding="utf-8", errors="ignore")).ids, dtype=torch.long)
        torch.save(ids, cache)
    print(f"tokens in corpus: {ids.numel():,}  vocab={cfg.vocab}")
    return ids


def batch_at(ids: torch.Tensor, cursor: int, batch: int, T: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    need = batch * T + 1
    if cursor + need >= ids.numel():
        cursor = 0
    window = ids[cursor : cursor + need]
    x = window[:-1].view(batch, T)
    y = window[1:].view(batch, T)
    return x, y, cursor + batch * T


def auxiliary_loss(model: GPT, alpha: float) -> torch.Tensor:
    """Switch load loss. f has no gradient. P is the mean router probability."""
    terms = []
    for moe in model.moe_modules():
        scores = moe._token_scores
        if scores is None or moe.last_load is None:
            continue
        p = scores.mean(dim=0)
        f = (moe.last_load / moe.last_load.sum()).detach()
        n = scores.shape[-1]
        terms.append(alpha * n * (f * p).sum())
    if not terms:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(terms).mean()


def train_phase(
    model: GPT,
    ids: torch.Tensor,
    *,
    phase: str,
    steps: int,
    batch: int,
    cursor: int,
    log_f,
    balancer: str = "bias",
    aux_alpha: float = 0.01,
    route: str = "hard",
    prob_steps: int = 0,
) -> dict:
    cfg = model.cfg
    model.train()
    for moe in model.moe_modules():
        moe.route = route
        if balancer != "bias":
            moe.bias_step = 0.0
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    losses = []
    t0 = time.perf_counter()
    tokens = 0
    for step in range(1, steps + 1):
        x, y, cursor = batch_at(ids, cursor, batch, cfg.T)
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        if prob_steps and step <= prob_steps:
            for moe in model.moe_modules():
                moe.route = "prob"
        elif prob_steps:
            for moe in model.moe_modules():
                moe.route = "hard"
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab), y.reshape(-1))
        if balancer == "aux":
            loss = loss + auxiliary_loss(model, aux_alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if balancer == "bias":
            for moe in model.moe_modules():
                moe.update_bias()
        tokens += batch * cfg.T
        value = float(loss.detach())
        losses.append(value)
        rec = {
            "phase": phase,
            "step": step,
            "loss": value,
            "tokens": tokens,
            "tok_s": tokens / max(time.perf_counter() - t0, 1e-9),
        }
        moes = model.moe_modules()
        if moes and moes[0].last_load is not None:
            load = torch.stack([m.last_load.cpu() for m in moes]).mean(0)
            rec["load"] = [round(float(v), 4) for v in load]
            rec["dead"] = int((load < 1e-6).sum())
            rec["max_vio"] = round(float((load.max() - load.mean()) / load.mean()), 4)
        log_f.write(json.dumps(rec) + "\n")
        if step == 1 or step % 50 == 0 or step == steps:
            extra = ""
            if "max_vio" in rec:
                extra = f"  dead={rec['dead']}  max_vio={rec['max_vio']}"
            print(f"  {phase:16} step {step:4d}/{steps}  loss={value:.3f}  tok/s={rec['tok_s']:.0f}{extra}")
    wall = time.perf_counter() - t0
    return {
        "phase": phase,
        "steps": steps,
        "tokens": tokens,
        "start_loss": losses[0],
        "end_loss": losses[-1],
        "min_loss": min(losses),
        "mean_last_50": sum(losses[-50:]) / min(50, len(losses)),
        "tok_s": tokens / wall,
        "wall_s": wall,
        "cursor": cursor,
        "curve": losses,
        "params": n_params(model),
    }


def active_params(model: GPT) -> int:
    """Weights a token actually multiplies. k experts, not all of them."""
    cfg = model.cfg
    total = 0
    for name, p in model.named_parameters():
        if ".experts." in name:
            continue
        total += p.numel()
    one = model.blocks[0].mlp
    if isinstance(one, MoE):
        expert_n = sum(p.numel() for p in one.experts[0].parameters())
        total += cfg.n_layer * cfg.k * expert_n
    return total


@torch.no_grad()
def forward_gap(dense: GPT, other: GPT, ids: torch.Tensor, batch: int) -> float:
    """Max |logit| gap on one batch. Identical experts must match the dense MLP."""
    x, _, _ = batch_at(ids, 0, batch, dense.cfg.T)
    x = x.to(DEVICE)
    dense.eval()
    other.eval()
    gap = float((dense(x) - other(x)).abs().max())
    dense.train()
    other.train()
    return gap


def plot(phases: list[dict]) -> None:
    FIGS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dense_n = phases[0]["steps"]
    for ph in phases:
        if ph["phase"] == "dense":
            xs = list(range(1, len(ph["curve"]) + 1))
        else:
            xs = list(range(dense_n + 1, dense_n + len(ph["curve"]) + 1))
        ax.plot(xs, ph["curve"], label=f"{ph['phase']}  {ph['start_loss']:.2f}→{ph['end_loss']:.2f}")
    ax.axvline(dense_n, color="0.5", ls="--", lw=0.8, label="upcycle")
    ax.set_xlabel("step (after the split, every arm sees the same next tokens)")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Dense linear FFN, then the same block as an MoE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "loss_phases.png", dpi=140)
    plt.close()

    loads = [ph for ph in phases if ph.get("final_load")]
    if not loads:
        return
    fig, ax = plt.subplots(figsize=(8, 3.6))
    experts = list(range(len(loads[0]["final_load"])))
    width = 0.8 / len(loads)
    for i, ph in enumerate(loads):
        ax.bar([e + i * width for e in experts], ph["final_load"], width=width, label=ph["phase"])
    k = loads[0].get("upcycle", {}).get("k", 2)
    n = len(experts)
    ax.axhline(k / n, color="0.4", ls="--", lw=0.8, label=f"even share {k}/{n}")
    ax.set_xlabel("expert")
    ax.set_ylabel("tokens routed, as a fraction of the batch")
    ax.set_title("Last-step load, averaged over layers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "expert_load.png", dpi=140)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-dense", type=int, default=400)
    ap.add_argument("--steps-after", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps_dense = 30
        args.steps_after = 30

    cfg = Cfg()
    seed_all(cfg.seed)
    print(f"device={DEVICE}  torch={torch.__version__}")
    check = router_example()
    ids = load_ids(cfg)
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / ("smoke.jsonl" if args.smoke else "train.jsonl")
    if log_path.exists():
        log_path.unlink()

    model = GPT(cfg).to(DEVICE)
    dense_params = n_params(model)
    print(f"dense params={dense_params:,}  active={active_params(model):,}")

    with log_path.open("w") as log_f:
        dense = train_phase(
            model, ids, phase="dense", steps=args.steps_dense, batch=args.batch, cursor=0, log_f=log_f
        )
        snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cursor = dense["cursor"]

        dense_go = train_phase(
            model, ids, phase="dense_continue", steps=args.steps_after, batch=args.batch, cursor=cursor, log_f=log_f
        )

        def arm(method: str) -> dict:
            seed_all(cfg.seed)
            fresh = GPT(cfg)
            fresh.load_state_dict(snap)
            info = upcycle(fresh, method)
            fresh.to(DEVICE)
            ref = GPT(cfg)
            ref.load_state_dict(snap)
            ref.to(DEVICE)
            info["max_abs_logit_gap_vs_dense"] = forward_gap(ref, fresh, ids, args.batch)
            del ref
            print(
                f"upcycled {method}: params={info['params']:,}  "
                f"active={active_params(fresh):,}  redraw/expert={info['neurons_redrawn_per_expert']}  "
                f"logit gap vs dense={info['max_abs_logit_gap_vs_dense']:.3e}"
            )
            rec = train_phase(
                fresh,
                ids,
                phase=f"moe_{method}",
                steps=args.steps_after,
                batch=args.batch,
                cursor=cursor,
                log_f=log_f,
            )
            rec["upcycle"] = info
            rec["active_params"] = active_params(fresh)
            moes = fresh.moe_modules()
            if moes and moes[0].last_load is not None:
                load = torch.stack([m.last_load.cpu() for m in moes]).mean(0)
                rec["final_load"] = [round(float(v), 4) for v in load]
            del fresh
            if DEVICE == "mps":
                torch.mps.empty_cache()
            return rec

        drop = arm("drop")
        copy = arm("copy")
        part = arm("partition")

    phases = [dense, dense_go, drop, copy, part]
    plot(phases)
    out = {
        "device": DEVICE,
        "torch": torch.__version__,
        "cfg": asdict(cfg),
        "router_check": check,
        "dense_params": dense_params,
        "phases": [{k: v for k, v in ph.items() if k != "curve"} | {"curve_len": len(ph["curve"])} for ph in phases],
        "curves": {ph["phase"]: ph["curve"] for ph in phases},
        "log": str(log_path.relative_to(ROOT)),
    }
    # keep full curves in results for the notebook; they are the plotted points, not every step
    path = ROOT / ("results_smoke.json" if args.smoke else "results.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    for ph in phases:
        print(
            f"  {ph['phase']:16} {ph['start_loss']:.3f} → {ph['end_loss']:.3f}  "
            f"mean_last_50={ph['mean_last_50']:.3f}  params={ph['params']:,}"
        )


if __name__ == "__main__":
    main()
