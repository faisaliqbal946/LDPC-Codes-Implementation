#!/usr/bin/env python3
"""Generate the LDPC experiment report.

The report focuses only on the large (204,104) LDPC experiment produced by
experiment.py. It reads figures from results/figures, numeric tables from
results/tables, and the JSON summary from results/summaries.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from fpdf import FPDF


REPORT_FILE = "report.pdf"
SUMMARY_PATH = Path("results/summaries/experiment_summary.json")
FIGURES_DIR = Path("results/figures")
TABLES_DIR = Path("results/tables")


FIGURE_PATHS = {
    "bsc": FIGURES_DIR / "bsc_ber_fer.png",
    "awgn": FIGURES_DIR / "awgn_ber_fer.png",
    "bec": FIGURES_DIR / "bec_ber_fer.png",
    "convergence": FIGURES_DIR / "decoder_convergence.png",
    "coding_gain": FIGURES_DIR / "coding_gain_summary.png",
}

LEGACY_FIGURE_FALLBACKS = {
    "bsc": FIGURES_DIR / "ber_bsc_with_capacity.png",
    "awgn": FIGURES_DIR / "ber_awgn_with_shannon.png",
    "bec": FIGURES_DIR / "ber_bec_with_capacity.png",
}

TABLE_PATHS = {
    "BSC": TABLES_DIR / "bsc_results.csv",
    "AWGN": TABLES_DIR / "awgn_results.csv",
    "BEC": TABLES_DIR / "bec_results.csv",
}

REFERENCES = [
    {
        "id": "[1]",
        "text": "Gallager, R. G., Low-Density Parity-Check Codes, MIT Press, 1963.",
        "url": "https://mitpress.mit.edu/9780262570940/low-density-parity-check-codes/",
    },
    {
        "id": "[2]",
        "text": "MacKay, D. J. C., Information Theory, Inference, and Learning Algorithms, Cambridge University Press, 2003.",
        "url": "https://www.inference.org.uk/itila/book.html",
    },
    {
        "id": "[3]",
        "text": "Richardson, T. and Urbanke, R., Modern Coding Theory, Cambridge University Press, 2008.",
        "url": "https://doi.org/10.1017/CBO9780511791338",
    },
    {
        "id": "[4]",
        "text": "Shannon, C. E., A Mathematical Theory of Communication, Bell System Technical Journal, 1948.",
        "url": "https://doi.org/10.1002/j.1538-7305.1948.tb01338.x",
    },
]


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "LDPC Codes Implementation - Large (204,104) Experiment", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110, 110, 110)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(25, 60, 105)
        self.multi_cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(190, 205, 225)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def subsection(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(45, 45, 45)
        self.multi_cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")

    def body(self, text: str):
        self.set_font("Helvetica", "", 10.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, clean_text(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def bullet_list(self, items: list[str]):
        self.set_font("Helvetica", "", 10.3)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.multi_cell(0, 5.2, clean_text(f"- {item}"), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def code(self, text: str):
        self.set_font("Courier", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(245, 247, 250)
        self.multi_cell(0, 5, clean_text(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def reference_list(self, refs: list[dict]):
        for ref in refs:
            self.set_font("Helvetica", "", 10.3)
            self.set_text_color(30, 30, 30)
            self.write(5.5, clean_text(f"{ref['id']} {ref['text']} "))
            self.set_text_color(20, 85, 150)
            try:
                self.write(5.5, clean_text(ref["url"]), link=ref["url"])
            except TypeError:
                self.write(5.5, clean_text(ref["url"]))
            self.ln(7)
        self.ln(1)


def clean_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u03b5": "epsilon",
        "\u2248": "~",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_table(rows: list[dict]) -> dict:
    by_decoder: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_decoder[row.get("decoder", "unknown")].append(row)

    out = {}
    for decoder, decoder_rows in by_decoder.items():
        gains = [as_float(r.get("coding_gain_dB")) for r in decoder_rows]
        gains = [g for g in gains if math.isfinite(g)]
        convergence = [as_float(r.get("convergence_rate")) for r in decoder_rows]
        convergence = [c for c in convergence if math.isfinite(c)]
        ber = [as_float(r.get("decoder_BER")) for r in decoder_rows]
        ber = [b for b in ber if math.isfinite(b)]
        fer = [as_float(r.get("decoder_FER")) for r in decoder_rows]
        fer = [v for v in fer if math.isfinite(v)]
        negative_gain = [r for r in decoder_rows if as_float(r.get("coding_gain_dB")) < 0]
        out[decoder] = {
            "points": len(decoder_rows),
            "min_gain": min(gains) if gains else math.nan,
            "max_gain": max(gains) if gains else math.nan,
            "negative_gain_points": len(negative_gain),
            "mean_convergence": sum(convergence) / len(convergence) if convergence else math.nan,
            "min_ber": min(ber) if ber else math.nan,
            "max_ber": max(ber) if ber else math.nan,
            "min_fer": min(fer) if fer else math.nan,
            "max_fer": max(fer) if fer else math.nan,
        }
    return out


def fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def summary_value(summary: dict, *keys, default="n/a"):
    for key in keys:
        if key in summary:
            return summary[key]
    return default


def add_title_page(pdf: ReportPDF, summary: dict):
    pdf.add_page()
    pdf.set_y(48)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(20, 55, 95)
    pdf.multi_cell(0, 12, "LDPC Codes Implementation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 15)
    pdf.set_text_color(45, 45, 45)
    pdf.multi_cell(0, 9, "Large (204,104) LDPC Experiment Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(14)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, "Course: Information Theory", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, "Author: Faisal Iqbal", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, "Language: Python", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(16)
    n = summary_value(summary, "N", default=204)
    k = summary_value(summary, "K", default=104)
    m = summary_value(summary, "M", default=100)
    rate = summary_value(summary, "RATE", "rate", default=0.5098)
    frames = summary_value(summary, "frames_per_point", default=300)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(235, 241, 248)
    pdf.cell(0, 9, clean_text(f"Experiment configuration: N={n}, K={k}, M={m}, R={fmt(as_float(rate))}, frames per point={frames}"), align="C", fill=True)


def add_artifact_note(pdf: ReportPDF):
    missing = []
    for path in FIGURE_PATHS.values():
        if not path.exists():
            missing.append(str(path))
    for path in TABLE_PATHS.values():
        if not path.exists():
            missing.append(str(path))
    if missing:
        pdf.subsection("Artifact Note")
        pdf.body(
            "Some expected experiment artifacts were not found when this report was generated. "
            "Run python experiment.py to regenerate the latest figures, CSV tables, and summary JSON. "
            "Missing artifacts:"
        )
        pdf.bullet_list(missing)


def add_figure(pdf: ReportPDF, title: str, path: Path, fallback: Path | None = None):
    actual = path if path.exists() else fallback
    pdf.subsection(title)
    if actual and actual.exists():
        max_width = pdf.w - pdf.l_margin - pdf.r_margin
        pdf.image(str(actual), w=max_width)
        if actual != path:
            pdf.body(f"Note: expected {path} was not available, so the existing figure {actual} was used as a fallback.")
    else:
        pdf.body(f"Figure not available: {path}. Run python experiment.py to generate it.")


def add_table_summary(pdf: ReportPDF, channel: str, rows: list[dict]):
    pdf.subsection(f"{channel} Numeric Summary")
    if not rows:
        pdf.body(f"No CSV table was available for {channel}. Expected file: {TABLE_PATHS[channel]}.")
        return

    stats = summarize_table(rows)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(229, 236, 245)
    widths = [28, 23, 23, 25, 26, 25, 25]
    headers = ["Decoder", "Min BER", "Max BER", "Max FER", "Min gain", "Max gain", "Mean conv."]
    for width, header in zip(widths, headers):
        pdf.cell(width, 7, header, border=1, align="C", fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8.3)
    for decoder, s in stats.items():
        values = [
            decoder,
            fmt(s["min_ber"], 3),
            fmt(s["max_ber"], 3),
            fmt(s["max_fer"], 3),
            fmt(s["min_gain"], 2),
            fmt(s["max_gain"], 2),
            fmt(s["mean_convergence"], 3),
        ]
        for width, value in zip(widths, values):
            pdf.cell(width, 6, clean_text(str(value)), border=1, align="C")
        pdf.ln()
    pdf.ln(2)

    negative = []
    for decoder, s in stats.items():
        if s["negative_gain_points"]:
            negative.append(f"{decoder}: {s['negative_gain_points']} point(s) with negative coding gain")
    if negative:
        pdf.body("Negative coding gain was observed and is reported directly rather than hidden:")
        pdf.bullet_list(negative)


def add_sections(pdf: ReportPDF, summary: dict, tables: dict[str, list[dict]]):
    n = int(summary_value(summary, "N", default=204))
    k = int(summary_value(summary, "K", default=104))
    m = int(summary_value(summary, "M", default=100))
    rate = as_float(summary_value(summary, "RATE", "rate", default=104 / 204))
    frames = int(summary_value(summary, "frames_per_point", default=300))
    max_iter_bp = int(summary_value(summary, "max_iter_bp_minsum", default=50))
    max_iter_bf = int(summary_value(summary, "max_iter_bit_flip", default=15))

    pdf.add_page()
    pdf.section("2. Abstract")
    pdf.body(
        f"This report presents a large-block LDPC experiment for a Gallager-style ({n},{k}) code [1] with "
        f"{m} parity checks and actual rate R={fmt(rate)}. The study evaluates transmission over the "
        "Binary Symmetric Channel, Additive White Gaussian Noise channel, and Binary Erasure Channel. "
        "Decoder performance is measured using bit error rate, frame error rate, convergence rate, "
        "average iteration count, and coding gain relative to uncoded baselines."
    )
    add_artifact_note(pdf)

    pdf.section("3. Introduction")
    pdf.body(
        "Low-Density Parity-Check codes were introduced by Gallager [1] as sparse linear block codes decoded by iterative message "
        "passing on a Tanner graph. Their practical value comes from sparse parity constraints and "
        "soft-information decoders such as Belief Propagation and Min-Sum, as commonly presented in coding theory texts [2], [3]. This report focuses on the "
        "large experiment used by the project workflow and does not include toy-code demonstrations."
    )

    pdf.section("4. LDPC Code Construction")
    diagnostics = summary.get("H_diagnostics", {})
    pdf.body(
        f"The experiment targets length 200 and resolves to the nearest valid Gallager construction [1] "
        f"length N={n}. The resulting parity-check matrix has M={m} rows and rank "
        f"{diagnostics.get('rank_gf2', m)} over GF(2), giving actual dimension K={k} and rate "
        f"{fmt(rate)}. The report does not print full H or G matrices; instead it summarizes their "
        "structural properties."
    )
    if diagnostics:
        pdf.bullet_list([
            f"Matrix density: {fmt(as_float(diagnostics.get('density')), 5)}",
            f"Row weight min/max/mean: {diagnostics.get('row_weight_min', 'n/a')} / {diagnostics.get('row_weight_max', 'n/a')} / {fmt(as_float(diagnostics.get('row_weight_mean')), 3)}",
            f"Column weight min/max/mean: {diagnostics.get('column_weight_min', 'n/a')} / {diagnostics.get('column_weight_max', 'n/a')} / {fmt(as_float(diagnostics.get('column_weight_mean')), 3)}",
            f"Estimated 4-cycles: {diagnostics.get('estimated_4_cycles', 'n/a')}",
        ])
    warnings = summary.get("diagnostics_warnings", [])
    if warnings:
        pdf.body("Construction diagnostics include the following warning(s):")
        pdf.bullet_list([str(w) for w in warnings])

    pdf.section("5. Encoding and Channel Models")
    pdf.body(
        "Encoding uses a generator matrix obtained from GF(2) row reduction of the parity-check matrix. "
        "Messages of length K are mapped to length-N codewords satisfying Hc = 0 over GF(2). The channel "
        "models are BSC with crossover probability p, AWGN with BPSK modulation and Eb/N0 sweeps, and BEC "
        "with erasure probability epsilon, following standard information-theoretic channel models [2], [4]. Uncoded baselines are included so that coding gain can be "
        "computed at each simulated point."
    )

    pdf.section("6. Decoding Algorithms")
    pdf.body(
        f"The BSC experiment evaluates Bit-Flip, Belief Propagation, and Min-Sum decoding [2], [3]. The AWGN "
        f"experiment evaluates BP and Min-Sum using soft LLRs. The BEC experiment evaluates BP and "
        f"Min-Sum only through the supported LLR decoder path; if their curves coincide, the report treats "
        f"that as an observed equality rather than as evidence of different behavior. BP and Min-Sum use "
        f"a maximum of {max_iter_bp} iterations, while Bit-Flip uses {max_iter_bf} iterations."
    )

    pdf.section("7. Experimental Methodology")
    pdf.body(
        f"Each channel point uses {frames} Monte Carlo frames by default. For every channel point and "
        "decoder, the experiment records uncoded BER, decoder BER, decoder FER, convergence rate, average "
        "iterations, and coding gain in dB. Coding gain is computed directly as "
        "10 log10(uncoded_BER / decoder_BER). The use of uncoded baselines follows the comparison style used in information theory and coding experiments [2], [4]. Negative values are retained because they indicate that the "
        "coded configuration performed worse than the uncoded baseline at that point."
    )
    pdf.code(
        "python experiment.py\n"
        "python generate_report.py"
    )

    pdf.section("8. Results and Discussion")
    pdf.subsection("BSC")
    add_figure(pdf, "BSC BER/FER", FIGURE_PATHS["bsc"], LEGACY_FIGURE_FALLBACKS.get("bsc"))
    add_table_summary(pdf, "BSC", tables["BSC"])
    pdf.body(
        "The BSC results compare the hard-decision Bit-Flip decoder with soft-decision BP and Min-Sum. "
        "The uncoded baseline is shown explicitly. The table should be used to identify any points where "
        "the coded BER is worse than uncoded, rather than assuming uniform improvement."
    )

    pdf.subsection("AWGN")
    add_figure(pdf, "AWGN BER/FER", FIGURE_PATHS["awgn"], LEGACY_FIGURE_FALLBACKS.get("awgn"))
    add_table_summary(pdf, "AWGN", tables["AWGN"])
    pdf.body(
        "The AWGN results use Eb/N0 as the sweep variable. Shannon references [4], when present in the figure, "
        "are theoretical markers only. They are not claims that the finite-length implementation approaches "
        "capacity. Negative coding gain can occur at low or poorly matched operating points."
    )

    pdf.subsection("BEC")
    add_figure(pdf, "BEC BER/FER", FIGURE_PATHS["bec"], LEGACY_FIGURE_FALLBACKS.get("bec"))
    add_table_summary(pdf, "BEC", tables["BEC"])
    pdf.body(
        "The BEC results should be interpreted carefully when BP and Min-Sum coincide. Matching values are "
        "reported as matching values and should not be described as separate performance behavior."
    )

    pdf.subsection("Coding Gain")
    add_figure(pdf, "Coding Gain Summary", FIGURE_PATHS["coding_gain"])
    pdf.body(
        "Coding gain is signed. Positive values indicate improvement relative to the uncoded baseline, "
        "while negative values indicate the coded decoder performed worse at that channel point. This report "
        "therefore describes each channel point according to the measured table values."
    )

    pdf.subsection("Convergence Behavior")
    add_figure(pdf, "Decoder Convergence", FIGURE_PATHS["convergence"])
    pdf.body(
        "Convergence rate and average iteration count provide context for BER and FER. A decoder may show "
        "acceptable BER at some points while still requiring many iterations or failing to converge often at "
        "more difficult channel settings."
    )

    validation_warnings = summary.get("validation_warnings") or summary.get("validation_issues") or []
    if validation_warnings:
        pdf.subsection("Validation Warnings")
        pdf.body("The experiment summary includes the following warnings. They are intentionally preserved:")
        pdf.bullet_list([str(w) for w in validation_warnings])

    pdf.section("9. Limitations")
    pdf.body(
        "The implementation is an educational Python implementation rather than an optimized LDPC simulator. "
        "The block length is finite, the construction may contain short cycles, and the default Monte Carlo "
        "budget is modest. Results therefore show the behavior of this specific code and decoder setup; they "
        "should not be generalized as near-capacity performance [3], [4]."
    )

    pdf.section("10. Conclusion")
    pdf.body(
        f"The project implements and evaluates a large ({n},{k}) LDPC code across BSC, AWGN, and BEC channel "
        "models. The current workflow records both reliability metrics and diagnostic metrics, including FER, "
        "convergence rate, average iterations, and signed coding gain. The most important interpretation is "
        "empirical: the generated CSV tables and summary JSON determine whether coding gain is positive or "
        "negative at each channel point."
    )

    pdf.section("11. References")
    pdf.reference_list(REFERENCES)


def generate():
    summary = load_summary()
    tables = {channel: load_table(path) for channel, path in TABLE_PATHS.items()}

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    add_title_page(pdf, summary)
    add_sections(pdf, summary, tables)
    pdf.output(REPORT_FILE)
    print(f"Report saved to: {REPORT_FILE}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    generate()
