"""レポート第2節用の概念図を自分で描く（論文の図の転載ではない）."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams["font.family"] = "Hiragino Sans"
plt.rcParams["axes.unicode_minus"] = False

OUT = os.environ.get("FIGDIR", os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(OUT, exist_ok=True)

C_DEC = "#4C78A8"      # 確定済み
C_NEW = "#F58518"      # このステップで確定
C_MASK = "#DDDDDD"     # マスク


def _cell(ax, i, j, color, text="", fs=9, ec="white"):
    ax.add_patch(Rectangle((i, -j), 0.92, 0.92, facecolor=color, edgecolor=ec, lw=1.2))
    if text:
        ax.add_patch(Rectangle((i, -j), 0.92, 0.92, facecolor="none", edgecolor="none"))
        ax.text(i + 0.46, -j + 0.46, text, ha="center", va="center", fontsize=fs,
                color="white" if color != C_MASK else "#555555")


def fig_trace_step():
    """図1: 何が trace step なのか（生成過程の格子表示）と収縮パラメータ s."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    L = 8
    # 各ステップで確定した位置（例）
    steps = [[2, 5], [0, 3], [1, 6, 7], [4]]

    ax = axes[0]
    decided = set()
    for t, pos in enumerate(steps):
        for i in range(L):
            if i in pos:
                _cell(ax, i, t, C_NEW, f"$o_{i}$")
            elif i in decided:
                _cell(ax, i, t, C_DEC)
            else:
                _cell(ax, i, t, C_MASK, "M")
        decided |= set(pos)
        ax.text(-0.9, -t + 0.46, f"$\\tau({t+1})$", ha="right", va="center", fontsize=11)
    for i in range(L):
        _cell(ax, i, len(steps), C_DEC)
    ax.text(-0.9, -len(steps) + 0.46, "完成", ha="right", va="center", fontsize=10)
    ax.set_title("(a) trace step = 1回のデノイジングで確定するトークンの集合", fontsize=11)
    ax.set_xlim(-3.2, L + 0.2)
    ax.set_ylim(-len(steps) - 0.2, 1.4)
    ax.axis("off")
    ax.text(L / 2, 1.0, "応答の位置 →", ha="center", fontsize=10)

    ax = axes[1]
    shrunk = [steps[0] + steps[1], steps[2] + steps[3]]
    decided = set()
    for t, pos in enumerate(shrunk):
        for i in range(L):
            if i in pos:
                _cell(ax, i, t, C_NEW, f"$o_{i}$")
            elif i in decided:
                _cell(ax, i, t, C_DEC)
            else:
                _cell(ax, i, t, C_MASK, "M")
        decided |= set(pos)
        ax.text(-0.9, -t + 0.46, f"$\\tau^s({t+1})$", ha="right", va="center", fontsize=11)
    ax.set_title("(b) 収縮パラメータ $s=2$：隣り合う2ステップを1つに束ねる", fontsize=11)
    ax.set_xlim(-3.2, L + 0.2)
    ax.set_ylim(-len(steps) - 0.2, 1.4)
    ax.axis("off")
    ax.text(L / 2, 1.0, "学習時の forward 回数が 1/s になる", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_trace_step.png"), dpi=180)
    plt.close(fig)


def fig_advantage_flow():
    """図2: トークン報酬 → ステップ量 → GAE → トークン advantage の流れ."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#444444", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2, text="", dy=0.25):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=14, lw=1.4, color="#444444"))
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", fontsize=9)

    box(0.3, 4.2, 2.6, 1.0, "トークン報酬 $r_j$\n(最終ステップのみ非零)", "#EAF0F6")
    box(0.3, 2.4, 2.6, 1.0, "トークン価値 $V_j^{old}$\n(価値モデルの出力)", "#EAF0F6")
    box(4.0, 3.3, 2.6, 1.0, "ステップ量へ平均\n$r^*_t,\\ V^{*,old}_t$", "#FCE7D2")
    box(7.4, 4.3, 2.4, 0.9, "TD 残差 $\\delta^*_t$", "#FCE7D2")
    box(7.4, 2.6, 2.4, 0.9, "ステップ GAE $A^*_t$", "#FCE7D2")
    box(4.0, 0.5, 5.8, 0.9, "トークン advantage "
        "$A_j=(r_j-V_j^{old})+\\gamma V^{*,old}_{t_j+1}+\\gamma\\lambda A^*_{t_j+1}$",
        "#DCEBDC")

    arrow(2.9, 4.7, 4.0, 4.0)
    arrow(2.9, 2.9, 4.0, 3.6)
    arrow(6.6, 4.0, 7.4, 4.7)
    ax.text(6.5, 4.95, "$\\delta^*_t=r^*_t-V^{*,old}_t+\\gamma V^{*,old}_{t+1}$",
            ha="right", fontsize=9)
    arrow(8.6, 4.3, 8.6, 3.5)
    ax.text(8.75, 3.85, "$A^*_t=\\delta^*_t+\\gamma\\lambda A^*_{t+1}$",
            ha="left", va="center", fontsize=9)
    arrow(8.6, 2.6, 8.0, 1.4)
    arrow(1.6, 2.4, 4.5, 1.4)

    ax.text(6.0, 5.6, "ステップ単位に一度集約してから、各トークンに配り戻す",
            ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_advantage_flow.png"), dpi=180)
    plt.close(fig)


def fig_objective_mismatch():
    """図3: ランダムマスク目的と実際の推論トラジェクトリのずれ."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    L = 8
    ax = axes[0]
    import random
    random.seed(3)
    for t, keep in enumerate([[0, 4, 6], [2, 3], [1, 5, 7, 0]]):
        for i in range(L):
            _cell(ax, i, t, C_DEC if i in keep else C_MASK, "" if i in keep else "M")
        ax.text(-0.6, -t + 0.46, f"標本{t+1}", ha="right", va="center", fontsize=10)
    ax.set_title("(a) ランダムマスク目的：見える位置は毎回でたらめ", fontsize=11)
    ax.set_xlim(-3.0, L + 0.2)
    ax.set_ylim(-3.2, 1.0)
    ax.axis("off")

    ax = axes[1]
    steps = [[0, 1], [2, 3], [4, 5]]
    decided = set()
    for t, pos in enumerate(steps):
        for i in range(L):
            if i in pos:
                _cell(ax, i, t, C_NEW)
            elif i in decided:
                _cell(ax, i, t, C_DEC)
            else:
                _cell(ax, i, t, C_MASK, "M")
        decided |= set(pos)
        ax.text(-0.6, -t + 0.46, f"step {t+1}", ha="right", va="center", fontsize=10)
    ax.set_title("(b) 実際の推論：左から順に確定していく", fontsize=11)
    ax.set_xlim(-3.0, L + 0.2)
    ax.set_ylim(-3.2, 1.0)
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_objective_mismatch.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    fig_trace_step()
    fig_advantage_flow()
    fig_objective_mismatch()
    print("saved to", os.path.abspath(OUT))
