"""results/results.json からレポート第5節用の図を作る."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("FIGDIR", os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results", "results.json")))

TASK_LABEL = {"sort": "sort（正解が1通り）", "branch": "branch（正解が2通り）"}
COL = {"sort": "#4C78A8", "branch": "#F58518"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180)
    plt.close(fig)
    print("saved", name)


def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, task in zip(axes, ("sort", "branch")):
        for obj, c in (("random", "#4C78A8"), ("semiar", "#54A24B"), ("trace", "#E45756")):
            xs = [p[0] for p in R["curves"][f"{task}_{obj}_s0"]]
            ys = [p[1] for p in R["curves"][f"{task}_{obj}_s0"]]
            ax.plot(xs, ys, label=obj, color=c)
        ax.set_yscale("log")
        ax.set_xlabel("学習ステップ")
        ax.set_ylabel("学習損失")
        ax.set_title(TASK_LABEL[task])
        ax.legend(fontsize=9)
    save(fig, "fig_exp_curves.png")


def fig_block():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax2 = ax.twinx()
    ax2.grid(False)
    for task in ("sort", "branch"):
        rows = R["sweep_block_1step"][task]
        b = [r["block_length"] for r in rows]
        a = [r["acc"] for r in rows]
        e = [r["acc_std"] for r in rows]
        ax.errorbar(b, a, yerr=e, marker="o", color=COL[task], label=TASK_LABEL[task])
        ax2.plot(b, [r["nfe"] for r in rows], ls="--", marker="x", color="gray", alpha=0.6)
    ax.set_xlabel("block_length（1ブロックを1ステップで確定）")
    ax.set_ylabel("完全一致率")
    ax2.set_ylabel("forward 回数（破線, 灰）")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1, 2, 3, 4, 6, 12])
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("ブロック長を伸ばすと速いが、正解が複数あるタスクでは壊れる")
    save(fig, "fig_exp_block.png")


def fig_steps():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for task in ("sort", "branch"):
        rows = R["sweep_steps"][task]
        s = [r["denoising_steps"] for r in rows]
        a = [r["acc"] for r in rows]
        e = [r["acc_std"] for r in rows]
        ax.errorbar(s, a, yerr=e, marker="o", color=COL[task], label=TASK_LABEL[task])
    ax.set_xlabel("denoising_steps（block_length=12 固定）")
    ax.set_ylabel("完全一致率")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([1, 2, 3, 4, 6, 12])
    ax.legend(fontsize=9)
    ax.set_title("(a) デノイジング回数と精度")

    ax = axes[1]
    for task in ("sort", "branch"):
        rows = R["sweep_steps"][task]
        ax.plot([r["nfe"] for r in rows], [r["acc"] for r in rows], marker="o",
                color=COL[task], label=TASK_LABEL[task])
        for r in rows:
            ax.annotate(str(r["denoising_steps"]), (r["nfe"], r["acc"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.set_xlabel("forward 回数（= 生成コスト）")
    ax.set_ylabel("完全一致率")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) コストと精度のトレードオフ（点の数字は denoising_steps）")
    ax.legend(fontsize=9)
    save(fig, "fig_exp_steps.png")


def fig_threshold():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, B in zip(axes, (4, 12)):
        ax2 = ax.twinx()
        ax2.grid(False)
        for task in ("sort", "branch"):
            rows = [r for r in R["sweep_threshold"][task] if r["block_length"] == B]
            t = [r["threshold"] for r in rows]
            ax.errorbar(t, [r["acc"] for r in rows], yerr=[r["acc_std"] for r in rows],
                        marker="o", color=COL[task], label=TASK_LABEL[task])
            ax2.plot(t, [r["nfe"] for r in rows], ls="--", marker="x", alpha=0.6,
                     color=COL[task])
        ax.set_xlabel("confidence_threshold")
        ax.set_ylabel("完全一致率（実線）")
        ax2.set_ylabel("forward 回数（破線）")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"block_length = {B}")
        if B == 4:
            ax.legend(fontsize=8, loc="center left")
    save(fig, "fig_exp_threshold.png")


def fig_trajectory():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.4))
    for col, task in enumerate(("sort", "branch")):
        tj = R["trajectory"][task]
        ax = axes[0][col]
        m = np.array(tj["example"])
        im = ax.imshow(m, aspect="auto", cmap="viridis")
        ax.set_title(f"{TASK_LABEL[task]}: static, 1トークン/ステップ", fontsize=10)
        ax.set_xlabel("応答の位置")
        ax.set_ylabel("テスト事例")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="確定ステップ")

        ax = axes[1][col]
        md = np.array(tj["dynamic_example"])
        im = ax.imshow(md, aspect="auto", cmap="viridis")
        ax.set_title(f"{TASK_LABEL[task]}: dynamic (T=0.9)", fontsize=10)
        ax.set_xlabel("応答の位置")
        ax.set_ylabel("テスト事例")
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="確定ステップ")
    save(fig, "fig_exp_trajectory.png")


def fig_trajectory_mean():
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for task in ("sort", "branch"):
        tj = R["trajectory"][task]
        ax.plot(range(1, 13), tj["mean_first_unmask"], marker="o", color=COL[task],
                label=f"{TASK_LABEL[task]} / static")
        ax.plot(range(1, 13), tj["dynamic_mean_first_unmask"], marker="s", ls="--",
                color=COL[task], alpha=0.6, label=f"{TASK_LABEL[task]} / dynamic")
    ax.set_xlabel("応答の位置")
    ax.set_ylabel("確定されたステップ番号の平均")
    ax.legend(fontsize=8)
    ax.set_title("どの位置が早く確定するか")
    save(fig, "fig_exp_trajectory_mean.png")


def fig_objective():
    settings = ["static_B4_s4", "static_B12_s3", "dynamic_B4_t0.9"]
    labels = ["static B=4, 4step", "static B=12, 3step", "dynamic B=4, T=0.9"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    width = 0.25
    for ax, task in zip(axes, ("sort", "branch")):
        for k, (obj, c) in enumerate((("random", "#4C78A8"), ("semiar", "#54A24B"),
                                      ("trace", "#E45756"))):
            means, errs = [], []
            for st in settings:
                runs = R["objective_table"][f"{task}|{obj}|{st}"]
                v = [r["acc"] for r in runs]
                means.append(float(np.mean(v)))
                errs.append(float(np.std(v)))
            ax.bar(np.arange(len(settings)) + (k - 1) * width, means, width,
                   yerr=errs, capsize=3, label=obj, color=c)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("完全一致率")
        ax.set_ylim(0, 1.05)
        ax.set_title(TASK_LABEL[task])
        ax.legend(fontsize=8)
    save(fig, "fig_exp_objective.png")


if __name__ == "__main__":
    fig_curves()
    fig_block()
    fig_steps()
    fig_threshold()
    fig_trajectory()
    fig_trajectory_mean()
    fig_objective()
