"""
Lecture checks on top of the dense → MoE conversion.

1. Hard top-k versus probabilistic selection, after cloning identical experts.
2. A second growth: 8 experts → 32, exact clones and half-redrawn clones.
3. No balancer, Switch auxiliary loss, and the bias used only for choosing.

Writes logs/extras.jsonl and results_extras.json. Does not touch the main 500-step run.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from moe_loop import (
    DEVICE,
    FIGS,
    LOGS,
    ROOT,
    Cfg,
    GPT,
    active_params,
    grow_experts,
    load_ids,
    n_params,
    seed_all,
    train_phase,
    upcycle,
)

STEPS_DENSE = 200
STEPS_BAL = 200
STEPS_GROW = 250
BATCH = 32
GROW_BATCH = 8


def tail(records: list[dict], key: str, n: int = 20) -> float:
    vals = [r[key] for r in records if key in r]
    if not vals:
        return 0.0
    chunk = vals[-n:]
    return sum(chunk) / len(chunk)


def dead_last(records: list[dict], n: int = 20) -> int:
    """Experts with zero mean load over the last n steps."""
    rows = [r["load"] for r in records if "load" in r][-n:]
    if not rows:
        return 0
    width = len(rows[0])
    dead = 0
    for e in range(width):
        if sum(row[e] for row in rows) <= 1e-8:
            dead += 1
    return dead


def run() -> None:
    cfg = Cfg()
    seed_all(cfg.seed)
    ids = load_ids(cfg)
    LOGS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    log_path = LOGS / "extras.jsonl"
    if log_path.exists():
        log_path.unlink()

    model = GPT(cfg).to(DEVICE)
    print(f"device={DEVICE}  dense params={n_params(model):,}")
    with log_path.open("w") as log_f:
        dense = train_phase(
            model, ids, phase="ex_dense", steps=STEPS_DENSE, batch=BATCH, cursor=0, log_f=log_f
        )
        cursor = dense["cursor"]
        up = upcycle(model, "copy")
        print(f"copied to {cfg.experts} experts  params={up['params']:,}")
        base = copy.deepcopy(model).cpu()

        def branch(name: str, *, build, **train) -> dict:
            fresh = copy.deepcopy(base).to(DEVICE)
            info = build(fresh)
            fresh.to(DEVICE)
            print(f"\n{name}  params={n_params(fresh):,}  active={active_params(fresh):,}  {info}")
            rec = train_phase(
                fresh,
                ids,
                phase=name,
                steps=train.pop("steps"),
                batch=train.pop("batch"),
                cursor=cursor,
                log_f=log_f,
                **train,
            )
            rec["build"] = info
            rec["active_params"] = active_params(fresh)
            del fresh
            if DEVICE == "mps":
                torch.mps.empty_cache()
            return rec

        bal_none = branch(
            "bal_none",
            build=lambda m: {"balancer": "none"},
            steps=STEPS_BAL,
            batch=BATCH,
            balancer="none",
        )
        bal_aux = branch(
            "bal_aux",
            build=lambda m: {"balancer": "aux", "alpha": 0.01},
            steps=STEPS_BAL,
            batch=BATCH,
            balancer="aux",
            aux_alpha=0.01,
        )
        bal_bias = branch(
            "bal_bias",
            build=lambda m: {"balancer": "bias", "gamma": cfg.bias_step},
            steps=STEPS_BAL,
            batch=BATCH,
            balancer="bias",
        )

        def grow_build(redraw: float):
            def _build(m: GPT) -> dict:
                return grow_experts(m, clones=4, redraw=redraw)

            return _build

        grow_hard = branch(
            "grow_hard",
            build=grow_build(0.0),
            steps=STEPS_GROW,
            batch=GROW_BATCH,
            balancer="none",
            route="hard",
        )
        grow_prob = branch(
            "grow_prob",
            build=grow_build(0.0),
            steps=STEPS_GROW,
            batch=GROW_BATCH,
            balancer="none",
            route="hard",
            prob_steps=STEPS_GROW // 2,
        )
        grow_drop = branch(
            "grow_drop",
            build=grow_build(0.5),
            steps=STEPS_GROW,
            batch=GROW_BATCH,
            balancer="bias",
            route="hard",
        )

    by_phase: dict[str, list[dict]] = {}
    for line in log_path.read_text().splitlines():
        rec = json.loads(line)
        by_phase.setdefault(rec["phase"], []).append(rec)

    arms = [bal_none, bal_aux, bal_bias, grow_hard, grow_prob, grow_drop]
    for arm in arms:
        rows = by_phase[arm["phase"]]
        arm["dead_last20"] = dead_last(rows)
        arm["mean_max_vio_last20"] = tail(rows, "max_vio")
        arm["mean_loss_last20"] = tail(rows, "loss")

    plot(by_phase, arms)
    slim = []
    for arm in [dense, *arms]:
        slim.append({k: v for k, v in arm.items() if k != "curve"})
    out = {
        "device": DEVICE,
        "steps_dense": STEPS_DENSE,
        "steps_bal": STEPS_BAL,
        "steps_grow": STEPS_GROW,
        "grow_batch": GROW_BATCH,
        "phases": slim,
    }
    path = ROOT / "results_extras.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    for arm in arms:
        print(
            f"  {arm['phase']:12} {arm['start_loss']:.3f} → {arm['end_loss']:.3f}  "
            f"mean20={arm['mean_loss_last20']:.3f}  dead_last20={arm['dead_last20']}  "
            f"max_vio20={arm['mean_max_vio_last20']:.3f}"
        )


def plot(by_phase: dict[str, list[dict]], arms: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name in ("bal_none", "bal_aux", "bal_bias"):
        ys = [r["loss"] for r in by_phase[name]]
        ax.plot(range(1, len(ys) + 1), ys, label=name)
    ax.set_xlabel("step after the copy")
    ax.set_ylabel("cross-entropy")
    ax.set_title("Same 8 experts, three balancers")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "balancer_loss.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name in ("grow_hard", "grow_prob", "grow_drop"):
        ys = [r.get("dead", 0) for r in by_phase[name]]
        ax.plot(range(1, len(ys) + 1), ys, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("dead experts (mean load across layers = 0)")
    ax.set_title("8 experts cloned to 32")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "grow_dead.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    run()
