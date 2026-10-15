"""Export the implemented SessionControl architecture as SVG, PDF, and PNG.

Use monochrome for inputs and existing AlphaEdit/LLM computation. A single teal
accent identifies the session-specific write protocol, delta bank, and router.
This is a conceptual illustration, not an experiment or an accuracy plot.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
INK = "#222222"
GRAY = "#606060"
LINE = "#868686"
PALE = "#F5F5F5"
TEAL = "#007A74"


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "sessioncontrol-architecture-v1",
        "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig = plt.figure(figsize=(18, 11.6), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1800), ylim=(1160, 0))
    ax.axis("off")

    def text(x, y, value, size=15, color=INK, weight="normal", align="center", **kw):
        return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                       ha=align, va="center", **kw)

    def box(x, y, width, height, contribution=False, fill="white", radius=13):
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            edgecolor=TEAL if contribution else LINE, facecolor=fill,
            linewidth=1.8 if contribution else 1.5,
            linestyle=(0, (5, 3)) if contribution else "solid",
        )
        ax.add_patch(patch)
        return patch

    def arrow(points, color=INK, width=1.8, dashed=False):
        if len(points) > 2:
            xs, ys = zip(*points[:-1])
            ax.plot(xs, ys, color=color, linewidth=width,
                    linestyle=(0, (4, 3)) if dashed else "solid")
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=18,
            color=color, linewidth=width, shrinkA=0, shrinkB=0,
            linestyle=(0, (4, 3)) if dashed else "solid",
        ))

    text(900, 52, "Session routing by delta activation", 30, weight="bold")
    text(900, 104, "SessionControl · implemented write → route → answer architecture", 16, GRAY)

    # Part A: the supplied original Q/A supervises one independent memory.
    text(65, 165, "(A)  Write and store session-specific deltas", 21, weight="bold", align="left")
    box(65, 215, 360, 240, fill=PALE)
    text(245, 246, "Session input", 18, weight="bold")
    text(90, 289, "Full answer-bearing session", 15, align="left")
    text(90, 329, r"Original question $q_s$", 15, align="left")
    text(90, 369, r"Supplied answer $a_s$", 15, align="left")
    text(245, 422, "Repeat independently for each session", 12.5, GRAY)

    box(500, 215, 560, 240, contribution=True)
    text(780, 245, "Session-specific memory write", 18, TEAL, "bold")
    box(521, 273, 518, 75, fill=PALE, radius=8)
    text(780, 299, "AlphaEdit grounded optimization", 16, weight="bold")
    text(780, 326, "FFN down-projections · layers 13–15", 13, GRAY)
    text(526, 376, "KEY", 11, GRAY, "bold", "left")
    text(578, 366, r"Q: $q_s$", 15, align="left")
    text(578, 392, "A:", 15, align="left")
    text(822, 376, "TARGET", 11, GRAY, "bold", "left")
    text(908, 376, r"$a_s$ + EOS", 15, align="left")
    text(780, 425, r"$\Delta W_s = W_{\mathrm{edited},s} - W_{\mathrm{base}}$", 18)
    text(780, 488, "Restore base weights after every write", 13, GRAY)
    arrow([(425, 335), (490, 335)])
    arrow([(1060, 335), (1174, 335)], TEAL)

    box(1185, 215, 550, 240, contribution=True)
    text(1460, 245, "Independent session delta bank", 18, TEAL, "bold")

    def matrix(x, y):
        # Three offsets symbolize the three stored layer matrices per delta.
        for offset in (10, 5, 0):
            ax.add_patch(Rectangle((x + offset, y - offset), 80, 70,
                                  facecolor="white", edgecolor=TEAL,
                                  linewidth=1.25))
        for dx in (80 / 3, 160 / 3):
            ax.plot([x + dx, x + dx], [y, y + 70], color=TEAL, linewidth=.8)
        for dy in (70 / 3, 140 / 3):
            ax.plot([x, x + 80], [y + dy, y + dy], color=TEAL, linewidth=.8)

    for x, label in [(1260, r"$\Delta W_1$"), (1410, r"$\Delta W_2$"), (1595, r"$\Delta W_N$")]:
        matrix(x, 303)
        text(x + 40, 400, label, 19, TEAL)
    text(1556, 339, "…", 23, TEAL)
    text(1460, 431, "Store each delta separately; keep the base model fixed", 12.5, GRAY)

    ax.plot([45, 1755], [522, 522], color="#C0C0C0", linewidth=1.2, linestyle=(0, (5, 4)))
    text(65, 565, "(B)  Route by activation, then generate the answer", 21, weight="bold", align="left")

    # Part B: unedited-model FFN inputs, not delta-generated text candidates.
    box(65, 658, 265, 180, fill=PALE)
    text(197.5, 691, r"Original question $q$", 17, weight="bold")
    text(197.5, 738, "Q: {question}", 14, family="DejaVu Sans Mono")
    text(197.5, 766, "A:", 14, family="DejaVu Sans Mono")
    text(197.5, 813, "Question only at recall", 12, GRAY)

    box(390, 658, 350, 180, fill=PALE)
    text(565, 691, "Base LLM forward pass", 17, weight="bold")
    text(565, 738, r"$W_{\mathrm{base}}$", 22)
    text(565, 780, "Capture FFN input activations", 13.5)
    text(565, 813, r"$k_\ell(q)$ at the final key token", 13, GRAY)
    arrow([(330, 747), (380, 747)])
    arrow([(740, 747), (801, 747)])
    text(772, 718, r"$k_\ell(q)$", 17)

    box(810, 620, 610, 290, contribution=True)
    text(1115, 653, "Delta activation router", 18, TEAL, "bold")
    text(1115, 691, "Score every stored delta using the same query activations", 13.5)
    text(1115, 752,
         r"$E_s(q)=\frac{\sum_{\ell\in L}\|\Delta W_{s,\ell}\,k_\ell(q)\|_2^2}"
         r"{\sum_{\ell\in L}\|\Delta W_{s,\ell}\|_F^2}$",
         19, TEAL)
    ax.plot([832, 1398], [810, 810], color=TEAL, linewidth=.8)
    text(1115, 847, r"Select $s^*=\arg\max_s E_s(q)$", 19, TEAL, "bold")
    text(1115, 888, "Equal scores: fixed lexicographic session-ID tie break", 12, GRAY)

    arrow([(1380, 455), (1380, 589), (1210, 589), (1210, 610)], TEAL)
    text(1542, 583, "Read one stored delta at a time", 13, TEAL)

    box(1490, 658, 265, 190, fill=PALE)
    text(1622.5, 691, "Answer LLM", 18, weight="bold")
    text(1622.5, 741, r"$W_{\mathrm{base}}+\Delta W_{s^*}$", 19)
    text(1622.5, 786, "Activate the selected delta", 12.5)
    text(1622.5, 820, "Greedy generation", 12.5, GRAY)
    arrow([(1420, 747), (1481, 747)], TEAL)
    text(1455, 718, r"$\Delta W_{s^*}$", 17, TEAL)

    # The same literal question reaches the final generator independently.
    arrow([(197.5, 838), (197.5, 942), (1450, 942), (1450, 803), (1481, 803)], GRAY, 1.6)
    text(835, 923, r"Same original question $q$", 14, GRAY)

    box(1490, 928, 265, 93)
    text(1622.5, 958, "Generated answer", 16, weight="bold")
    text(1622.5, 990, "From the selected memory", 12.5, GRAY)
    arrow([(1622.5, 848), (1622.5, 920)])
    text(1622.5, 887, "EOS / next-turn stop", 12.5, GRAY,
         bbox=dict(facecolor="white", edgecolor="none", pad=4))
    text(1115, 979, r"$L=\{13,14,15\}$ · normalized delta activation energy", 13, GRAY)
    text(780, 1015, "Restore base weights after answer generation", 12.5, GRAY)

    text(900, 1067,
         "Exact-question experiment: recall uses the question supplied at write time; one Q/A is supervised per session.",
         13, GRAY)
    ax.plot([470, 505], [1120, 1120], color=TEAL, linewidth=3)
    text(518, 1120, "Our session-memory protocol and activation routing", 12.5, TEAL, align="left")
    ax.plot([1116, 1151], [1120, 1120], color=GRAY, linewidth=3)
    text(1164, 1120, "Existing inputs, AlphaEdit, and LLM computation", 12.5, GRAY, align="left")

    out = ROOT / "docs/figures"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        target = out / f"sessioncontrol-architecture.{extension}"
        metadata = {"Title": "SessionControl architecture"}
        if extension == "svg":
            metadata["Date"] = None
        if extension == "pdf":
            metadata.update(Creator="SessionControl", CreationDate=None, ModDate=None)
        fig.savefig(target, dpi=180, facecolor="white", metadata=metadata)
        if extension == "svg":
            # Remove serializer-only whitespace without changing vector geometry.
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
        print(target.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
