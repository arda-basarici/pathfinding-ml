"""generate_report.py — build the PDF report "Learning to Guess the Distance".

Run from pathfinding-ml/:  python generate_report.py
Regenerates the report's experiments (cached under .report_cache/), renders print-quality
figures, and assembles a narrative PDF at the project root. The report is its own piece of
writing — a journey, not a dump of results — and every number is reproducible from the
seeds in report_data.
"""

from __future__ import annotations

import os
import tempfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Flowable, Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

import report_charts as rc
import report_data

ACCENT = colors.HexColor("#C0392B")
INK = colors.HexColor("#2B2B2B")
GREYDARK = colors.HexColor("#7F8C8D")
GREY = colors.HexColor("#BDC3C7")
GREYLITE = colors.HexColor("#ECF0F1")
WHITE = colors.white
OUT_PDF = "pathfinding_report.pdf"


# ---------------------------------------------------------------- styles
def styles():
    s = getSampleStyleSheet()
    out = {}
    out["title"] = ParagraphStyle("title", parent=s["Title"], fontName="Helvetica-Bold",
        fontSize=30, leading=34, textColor=INK, spaceAfter=6, alignment=TA_LEFT)
    out["subtitle"] = ParagraphStyle("subtitle", parent=s["Normal"], fontName="Helvetica",
        fontSize=14, leading=19, textColor=GREYDARK, spaceAfter=22, alignment=TA_LEFT)
    out["h2"] = ParagraphStyle("h2", parent=s["Heading2"], fontName="Helvetica-Bold",
        fontSize=17, leading=21, textColor=INK, spaceBefore=14, spaceAfter=2)
    out["kicker"] = ParagraphStyle("kicker", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=10, leading=13, textColor=ACCENT, spaceAfter=2, alignment=TA_LEFT)
    out["body"] = ParagraphStyle("body", parent=s["Normal"], fontName="Helvetica",
        fontSize=10.7, leading=15.5, textColor=INK, spaceAfter=9, alignment=TA_LEFT)
    out["thesis"] = ParagraphStyle("thesis", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=12, leading=17, textColor=ACCENT, spaceBefore=4, spaceAfter=12, leftIndent=10)
    out["caption"] = ParagraphStyle("caption", parent=s["Normal"], fontName="Helvetica-Oblique",
        fontSize=8.5, leading=11, textColor=GREYDARK, spaceBefore=2, spaceAfter=16, alignment=TA_CENTER)
    out["cover_tag"] = ParagraphStyle("cover_tag", parent=s["Normal"], fontName="Helvetica",
        fontSize=11, leading=17, textColor=INK, spaceAfter=4)
    out["foot"] = ParagraphStyle("foot", parent=s["Normal"], fontName="Helvetica",
        fontSize=9, leading=13, textColor=GREYDARK)
    out["tbl_h"] = ParagraphStyle("tbl_h", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=9.5, leading=12, textColor=WHITE)
    out["tbl_c"] = ParagraphStyle("tbl_c", parent=s["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=12.5, textColor=INK)
    out["tbl_cb"] = ParagraphStyle("tbl_cb", parent=s["Normal"], fontName="Helvetica-Bold",
        fontSize=9.5, leading=12.5, textColor=INK)
    return out


class HRule(Flowable):
    def __init__(self, width, color=GREY, thickness=0.8):
        super().__init__(); self.width = width; self.color = color; self.thickness = thickness
    def draw(self):
        self.canv.setStrokeColor(self.color); self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, self.width, 0)


class AccentBand(Flowable):
    def __init__(self, width=2.2 * cm, gap_above=5, gap_below=4, thickness=3):
        super().__init__()
        self.width = width; self.thickness = thickness
        self.gap_above = gap_above; self.gap_below = gap_below
        self.height = thickness + gap_above + gap_below
    def wrap(self, aw, ah):
        return (self.width, self.height)
    def draw(self):
        self.canv.setStrokeColor(ACCENT); self.canv.setLineWidth(self.thickness)
        y = self.gap_below + self.thickness / 2
        self.canv.line(0, y, self.width, y)


# ---------------------------------------------------------------- facts
def _win(rec):
    s = rec["axis2_summary"]
    return (s["astar+learned"]["mean_nodes_expanded"] / s["astar+manhattan"]["mean_nodes_expanded"] - 1) * 100
def _gap(rec):
    return rec["axis2_summary"]["astar+learned"]["mean_optimality_gap"] * 100
def _opt(rec):
    return rec["gap_distribution"]["astar+learned"]["frac_optimal"] * 100
def _swin(rec, style):
    s = rec["by_style"][style]
    return (s["astar+learned"]["mean_nodes_expanded"] / s["astar+manhattan"]["mean_nodes_expanded"] - 1) * 100
def _sgap(rec, style):
    return rec["by_style"][style]["astar+learned"]["mean_optimality_gap"] * 100
def _mae(rec):
    return rec["axis1_quality"]["learned"]["mae"]
def _over(rec):
    return rec["axis1_quality"]["learned"]["frac_overestimated"] * 100
def _imp(rec, feat):
    for d in rec["feature_importance"]:
        if d["feature"] == feat:
            return d["importance_mean"]
    return 0.0


def facts(data):
    b, g = data["baseline"], data["global"]
    return {
        "g_win_abs": sorted(abs(_win(r)) for r in data["global_seeds"]),
        "g_gap_max": max(_gap(r) for r in data["global_seeds"]),
        "b_win": _win(b), "b_gap": _gap(b),
        "b_scat_gap": _sgap(b, "scattered"), "b_struct_win": _swin(b, "structured"),
        "g_win": _win(g), "g_gap": _gap(g), "g_opt": _opt(g),
        "g_scat_win": _swin(g, "scattered"), "g_struct_win": _swin(g, "structured"),
        "ps_win": _win(data["pure_scattered"]), "ps_mae": _mae(data["pure_scattered"]),
        "pt_win": _win(data["pure_structured"]), "pt_mae": _mae(data["pure_structured"]),
        "min3_win": _win(data["min3"]),
        "s2t_win": _win(data["transfer_s2t"]), "s2t_gap": _gap(data["transfer_s2t"]), "s2t_over": _over(data["transfer_s2t"]),
        "t2s_win": _win(data["transfer_t2s"]), "t2s_gap": _gap(data["transfer_t2s"]), "t2s_over": _over(data["transfer_t2s"]),
        "imp_eucl": _imp(b, "euclidean_to_goal"), "imp_global": _imp(g, "global_obstacle_density"),
        "n": b["config"]["n"],
    }


# ---------------------------------------------------------------- build
def build(data, tmpdir):
    S = styles()
    PW = A4[0] - 4 * cm
    story = []
    P = lambda t, st="body": story.append(Paragraph(t, S[st]))
    SP = lambda h: story.append(Spacer(1, h))
    band = lambda w=2.2 * cm: story.append(AccentBand(width=w))
    f = facts(data)

    def figure(fn, name, caption, w=PW):
        path = os.path.join(tmpdir, name); fn(data, path)
        iw, ih = ImageReader(path).getSize()
        story.append(Image(path, width=w, height=w * ih / iw))
        story.append(Paragraph(caption, S["caption"]))

    # ===== COVER =====
    SP(2.2 * cm); band(3.2 * cm); SP(14)
    P("Learning to Guess<br/>the Distance", "title")
    P("A learned heuristic for A* — and what it took to know whether it actually helped", "subtitle")
    SP(6)
    P("A* finds shortest paths fast when its heuristic — its guess of the distance still to go — "
      "is good. Manhattan distance is the classic guess, but it's blind to walls. Could a model "
      "learn a better one? The surprising part of this project isn't the model. It's how much work "
      "it took to <i>know</i> whether the model helped at all — and how the answer kept changing as "
      "we looked closer.", "cover_tag")
    SP(18)
    for tag, txt in [
        ("THE AVERAGE LIES", "Pooled over all mazes the model looked like a wash — until we stopped pooling."),
        ("ONE FEATURE FLIPS IT", "A single global feature turned the wash into a clear, optimality-safe win."),
        ("IT WAS NEVER THE FEATURE", "The real lever was the training distribution; the feature just told the model which world it was in."),
        ("THE MODEL FORGETS OFF-DISTRIBUTION", "Train on one kind of maze, test on another, and it fails — in two opposite ways."),
    ]:
        story.append(Paragraph(tag, S["kicker"]))
        story.append(Paragraph(txt, S["body"]))
    SP(14); story.append(HRule(PW)); SP(6)
    P("github.com/arda-basarici/ai-journey &nbsp;·&nbsp; Phase 2 · all results regenerated from seeded runs", "foot")
    story.append(PageBreak())

    # ===== EXEC SUMMARY =====
    P("Did the model actually help?", "h2"); band(); SP(6)
    P("It is an easy question to answer badly. Train a model to predict cost-to-go, plug it into A* "
      "as the heuristic, measure the nodes it expands, and report a number. We did that, and the "
      "first number said: barely. The honest work was everything that came after — because the "
      "single pooled number turned out to be hiding the truth, twice over.")
    P("Two axes, never one. A heuristic is good only if it cuts the work A* does (<b>nodes "
      "expanded</b>) <i>without</i> sending it down longer paths (the <b>optimality gap</b>). "
      "Collapsing those into one score is how you fool yourself. Every claim here is also checked "
      "<i>within</i> maze type, not just pooled — the same discipline that keeps an average from "
      "lying.")
    SP(6)
    rows = [
        [Paragraph("The turn", S["tbl_h"]), Paragraph("What it revealed", S["tbl_h"])],
        [Paragraph("<b>The average lies</b>", S["tbl_cb"]),
         Paragraph(f"Pooled, the model was a wash (~{abs(f['b_win']):.0f}% fewer nodes at a "
                   f"{f['b_gap']:.1f}% optimality cost). Split by maze type it reversed: a clean win on "
                   f"corridors, a loss on open fields. A Simpson's-paradox trap.", S["tbl_c"])],
        [Paragraph("<b>One feature flips it</b>", S["tbl_cb"]),
         Paragraph(f"Adding global obstacle density: <b>{abs(f['g_win']):.0f}% fewer nodes at a "
                   f"{f['g_gap']:.1f}% gap</b>, {f['g_opt']:.0f}% of mazes solved optimally.", S["tbl_c"])],
        [Paragraph("<b>It's the data, not the feature</b>", S["tbl_cb"]),
         Paragraph(f"Trained on open fields alone, the plain model already wins ({abs(f['ps_win']):.0f}%). "
                   f"Global density is a <i>regime tag</i> that lets one model serve a mixed distribution — "
                   f"not a magic calibrator.", S["tbl_c"])],
        [Paragraph("<b>It forgets off-distribution</b>", S["tbl_cb"]),
         Paragraph(f"Train on one maze type, test on the other, and it fails in opposite ways: too timid "
                   f"({f['s2t_win']:+.0f}% nodes) one direction, too reckless ({f['t2s_gap']:.0f}% gap) the "
                   f"other.", S["tbl_c"])],
    ]
    t = Table(rows, colWidths=[PW * 0.30, PW * 0.70])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, GREYLITE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, GREY),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ===== §1 SETUP =====
    P("1 · A heuristic worth learning", "h2"); band(); SP(6)
    P("Of all the parts of A*, only one is a guess. The path is computed; the cost-so-far is known; "
      "but the heuristic — the estimate of remaining distance — is, by design, an approximation. "
      "That is exactly where machine learning belongs: predict the true cost-to-go from cheap "
      "features, and use the prediction as the heuristic. The true cost-to-go itself we get for free "
      "and exactly, by a single backward breadth-first sweep from each goal — perfect labels, so any "
      "error is the model's, not the data's.")
    P("The honest scaffolding matters as much as the model. We split the data by <i>whole maze</i>, "
      "never by cell, because cells in one maze are correlated — a random split would leak the test "
      "answers through their neighbours (and the code asserts the split is clean). The baseline isn't "
      "a strawman: it's the Manhattan heuristic A* already uses, which is admissible and good. And "
      "we judge on two axes — nodes expanded and optimality gap — reported within maze type. "
      "Two environments appear throughout: open fields of scattered obstacles, and structured "
      "corridor mazes.")
    P("Two scoping notes up front, in fairness to the reader. Everything here is simulation — two maze "
      "generators only (open scattered fields and perfect corridor mazes) on a 4-connected unit grid — so "
      "the findings describe this controlled world, not real road networks. And the model is deliberately "
      "simple: a gradient-boosted regressor on a handful of features. This is a literacy-phase project; its "
      "contribution is the evaluation discipline, not model sophistication.")
    story.append(Paragraph("A heuristic is only worth learning if it beats a strong baseline on the metric you actually care about — and proves it on data it never saw.", S["thesis"]))
    story.append(PageBreak())

    # ===== §2 WITHIN-GROUP =====
    P("2 · The average lies", "h2"); band(); SP(6)
    P(f"The first end-to-end result was deflating. Pooled over every held-out maze, the learned "
      f"heuristic expanded about {abs(f['b_win']):.0f}% fewer nodes than Manhattan — but paid a "
      f"{f['b_gap']:.1f}% optimality gap for it. A wash with a cost. If we had stopped there, the "
      f"verdict would have been: it didn't work.")
    figure(rc.fig_within_group, "f_within.png",
           "Six-feature model. Pooled, learned and Manhattan look similar; split by maze type, the "
           "story splits with them.")
    P(f"Then we refused to pool. Inside <i>open fields</i>, the model was actually worse — it gave up "
      f"~{f['b_scat_gap']:.0f}% optimality for no real speed gain, because Manhattan is already nearly "
      f"exact in open space. Inside <i>corridor mazes</i>, it was a clean win — about "
      f"{abs(f['b_struct_win']):.0f}% fewer nodes at no optimality cost, because Manhattan is a poor "
      f"guide where walls force long detours. Two opposite effects that cancel in the average.")
    story.append(Paragraph("Pool two opposite effects and the average reports that nothing happened. The truth lives within the groups.", S["thesis"]))
    P("This is Simpson's paradox in miniature, and it is the spine of the whole report: a pooled "
      "metric can not only blur a result but invert it. Every number that follows is read within "
      "maze type first.")
    story.append(PageBreak())

    # ===== §3 GLOBAL =====
    P("3 · One feature, and the verdict flips", "h2"); band(); SP(6)
    P("If the model helps where Manhattan is weak and hurts where it is already strong, the missing "
      "piece is something that tells it which situation it is in. We added one feature: the maze's "
      "overall obstacle density.")
    figure(rc.fig_global_fix, "f_global.png",
           "Adding global obstacle density: a much larger node reduction, and the optimality gap "
           "nearly vanishes.")
    P(f"The verdict flipped. The learned heuristic now expands about <b>{abs(f['g_win']):.0f}% fewer "
      f"nodes than Manhattan at a {f['g_gap']:.1f}% optimality gap</b>, solving {f['g_opt']:.0f}% of "
      f"held-out mazes on the exact shortest path. The open-field regime, previously a loss, turned into a "
      f"~{abs(f['g_scat_win']):.0f}% win — and that one is genuinely earned: open fields have many "
      f"competing paths, so its near-zero gap is real. (The corridor 0% gap is partly free — perfect "
      f"mazes have a single path, so any route found there is optimal — a point we return to in the limits.)")
    figure(rc.fig_seed_robustness, "f_seeds.png",
           "Each dot is one random seed; the win clusters tightly — not a one-seed fluke.", w=PW * 0.72)
    P(f"And it is not one lucky run. Across three random seeds the +global model expands between "
      f"{f['g_win_abs'][0]:.0f}% and {f['g_win_abs'][-1]:.0f}% fewer nodes, with the optimality gap never "
      f"above {f['g_gap_max']:.1f}%, and the result holds at 10,000 mazes. The six-feature baseline is "
      f"just as consistently a near-wash. The effect is the feature, not the seed.")
    story.append(Paragraph("A learned heuristic can beat a strong, admissible baseline on real search — given the one piece of context it was missing.", S["thesis"]))
    story.append(PageBreak())

    # ===== §4 THE REFRAME =====
    P("4 · It was never the feature — it was the data", "h2"); band(); SP(6)
    P("The tidy story would end there: we found a good feature. But asking <i>why</i> it worked "
      "overturned that story — and this is the most important page in the report.")
    figure(rc.fig_importance, "f_importance.png",
           "Permutation importance. With global density present, the local-clutter features the model "
           "leaned on before collapse to near-zero — it subsumes them.")
    P(f"Two clues. First, importance: global density dominates everything, and the local-clutter "
      f"features the six-feature model relied on drop to ~zero — it didn't <i>add</i> information so "
      f"much as <i>replace</i> a noisy proxy for it. Second, and decisively: trained on open fields "
      f"<i>alone</i>, the plain six-feature model already crushes them — about {abs(f['ps_win']):.0f}% "
      f"fewer nodes, with a tiny error (MAE {f['ps_mae']:.1f}) and <i>no global density at all</i>. "
      f"Trained on corridors alone it also does well (MAE {f['pt_mae']:.0f}). Global density adds almost "
      f"nothing in either pure case.")
    P("So global density is not an open-field calibrator. It is a <b>regime tag</b>. The earlier "
      "failure was never intrinsic to open fields — it happened only when one model was trained on a "
      "<i>mixture</i> of both maze types and had no way to tell them apart, so it compromised, and the "
      "open-field regime lost the compromise. Give it a feature that identifies the regime and the "
      "compromise dissolves.")
    story.append(Paragraph("The result was governed by the training distribution, not the model. Heterogeneous data needs a group-identifying feature — or a model per group.", S["thesis"]))
    story.append(PageBreak())

    # ===== §5 MINIMAL =====
    P("5 · How little it takes", "h2"); band(); SP(6)
    P(f"If global density subsumes the clutter features, most of the seven should be removable. They "
      f"are. A model with just three features — Manhattan and Euclidean distance plus global density — "
      f"matches the full set (about {abs(f['min3_win']):.0f}% fewer nodes), and prediction error stops "
      f"improving past three. The local-density and dead-end features were pure redundancy.")
    figure(rc.fig_minimal, "f_minimal.png",
           "Two-to-three features reach the full model's search performance; the rest add nothing.",
           w=PW * 0.74)
    story.append(Paragraph("Parsimony, earned empirically: the smallest honest model is the one that says everything the big one did.", S["thesis"]))
    story.append(PageBreak())

    # ===== §6 TRANSFER =====
    P("6 · The model only knows what it trained on", "h2"); band(); SP(6)
    P("If the training distribution is the real lever, the sharpest test is to break it on purpose: "
      "train on one maze type and test on the other. It fails completely — and, tellingly, in two "
      "<i>opposite</i> ways.")
    figure(rc.fig_transfer, "f_transfer.png",
           "Cross-distribution transfer. Trained on one regime, tested on the other, the heuristic "
           "either goes limp or goes reckless.")
    P(f"Trained on open fields and tested on corridors, it <i>under</i>-estimates the long detours "
      f"(it overestimates only {f['s2t_over']:.0f}% of cells), so the heuristic goes limp — A* expands "
      f"about as many nodes as plain Manhattan ({f['s2t_win']:+.0f}%), drifting toward blind search, but "
      f"stays optimal. Trained on corridors and tested on open fields, it <i>over</i>-estimates almost "
      f"everywhere ({f['t2s_over']:.0f}% of cells), turning A* greedy: {abs(f['t2s_win']):.0f}% fewer nodes "
      f"but a {f['t2s_gap']:.0f}% optimality gap — fast and wrong. Same model, same features; only the "
      f"training distribution changed.")
    P("A quiet signature confirms it: when tested off-distribution, the features' importance collapses "
      "toward zero — the model has stopped responding to its inputs, emitting a mis-scaled answer it "
      "learned elsewhere. (And global density doesn't rescue the transfer: the test density falls "
      "outside its training range, so it is extrapolating.) This is distribution shift made visible.")
    story.append(Paragraph("A learned heuristic is only as good as the distribution it learned. Move the world and it doesn't adapt — it mis-remembers.", S["thesis"]))
    story.append(PageBreak())

    # ===== §7 OPEN QUESTIONS =====
    P("What we didn't claim", "h2"); band(); SP(6)
    P("A few honest limits, left open on purpose — each a real next step rather than a hidden flaw:")
    P("<b>The corridor optimality is partly free.</b> Our structured mazes are <i>perfect</i> mazes — "
      "exactly one path between any two cells — so any path found there is optimal regardless of the "
      "heuristic. The defensible corridor win is the node reduction; whether an inadmissible learned "
      "heuristic stays optimal once mazes have loops (braided mazes) is untested, and is the first "
      "experiment we'd run next.")
    P("<b>The metric you optimise is not the metric you want.</b> The model was trained to minimise "
      "prediction error, but A* cares about ranking, not absolute accuracy — which is why a model with "
      "a large MAE on corridors still guides search well, and why we always checked the search "
      "behaviour, not just the error. Two layers, never conflated.")
    P("<b>Cheap features have a ceiling.</b> A regional feature (walls on the straight line to the "
      "goal) added nothing once global density was present, and corridor routing is fundamentally a "
      "<i>global</i> property — whether a passage dead-ends depends on far-away structure a cheap "
      "feature can't see. Pushing past this likely means feeding the whole maze to the model "
      "(Neural-A*-style), which is a different, heavier project.")
    P("<b>Admissibility is recoverable.</b> The learned heuristic is inadmissible by default; the code "
      "already has a transform hook to scale predictions down. Whether an admissibility-constrained "
      "model keeps the speed win is an open, answerable question.")
    story.append(PageBreak())

    # ===== CLOSE =====
    P("What the project is really about", "h2"); band(); SP(6)
    P("Set the turns side by side and the subject isn't really pathfinding. It's the discipline of "
      "knowing whether a result is real: refusing to trust a pooled average (it inverted on us); "
      "beating a strong baseline rather than a strawman; separating the metric you can optimise from "
      "the objective you care about; and — the turn that mattered most — being willing to overturn "
      "your own explanation when the data demands it. “We found a good feature” became "
      "“the training distribution was the lever all along.”")
    P("That last lesson is the one worth carrying: a model is a compression of the distribution it was "
      "trained on, and it is only trustworthy inside that distribution. The most useful thing this "
      "project produced is not a faster heuristic — it is a small, reproducible apparatus for asking, "
      "honestly, whether a model helped at all.")
    story.append(Paragraph("The hard part was never training the model. It was earning the right to say it worked.", S["thesis"]))
    SP(10); story.append(HRule(PW)); SP(6)
    P("Principles in play: Simpson's paradox; baseline discipline; data leakage and group-wise "
      "splitting; permutation importance (and its correlated-feature caveat); mixture-of-distributions; "
      "distribution shift / out-of-distribution generalisation; admissibility (Hart, Nilsson &amp; "
      "Raphael, 1968); metric–objective alignment. &nbsp;·&nbsp; Code, tests, and the full "
      "decision log: github.com/arda-basarici/ai-journey", "foot")

    return story


def main():
    print("gathering experiment results (first run computes + caches; later runs are instant)...")
    data = report_data.gather()
    with tempfile.TemporaryDirectory() as tmp:
        doc = SimpleDocTemplate(
            OUT_PDF, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
            title="Learning to Guess the Distance", author="arda-basarici",
        )
        doc.build(build(data, tmp))
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
