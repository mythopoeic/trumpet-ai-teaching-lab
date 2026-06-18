"""Generate a markdown evaluation report from scored results.

Run from teaching-engine/ directory:
    python tests/evaluation/generate_report.py --scored-file results/scored_2026-02-16_120000.json
    python tests/evaluation/generate_report.py --scored-file results/era_check_2026-02-16_120000.json
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from statistics import mean

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

SCORE_DIMS = ["factual_accuracy", "era_alignment", "completeness", "helpfulness"]
PASS_THRESHOLD = 3.5


def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def is_era_check_file(data):
    """Detect if this is a lightweight era_check JSON from --quick mode."""
    mode = data.get("metadata", {}).get("mode")
    if mode == "quick":
        return True
    # Also check if results lack bot_response (era_check format)
    if data.get("results") and "bot_response" not in data["results"][0]:
        return True
    return False


def avg_score(result):
    """Compute mean of the 4 numeric score dimensions for a result."""
    scores = result.get("scores")
    if not scores:
        return None
    return mean(scores[dim] for dim in SCORE_DIMS)


def generate_era_only_report(data):
    """Generate an era-detection-only report from era_check JSON."""
    meta = data["metadata"]
    results = data["results"]
    total = meta.get("total_questions", len(results))

    lines = []
    lines.append("# Era Detection Report")
    lines.append("")
    lines.append(f"**Date:** {meta.get('date', 'N/A')}")
    lines.append(f"**Base URL:** {meta.get('base_url', 'N/A')}")
    lines.append(f"**Mode:** Quick (era detection only)")
    lines.append("")

    # Era detection accuracy
    era_correct_count = sum(1 for r in results if r.get("era_correct"))
    era_pct = (era_correct_count / total * 100) if total > 0 else 0

    lines.append("## Era Detection Accuracy")
    lines.append("")
    lines.append(f"**Overall:** {era_correct_count}/{total} correct ({era_pct:.1f}%)")
    lines.append("")

    # Per-era accuracy
    era_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        era = r.get("expected_era", r.get("era", "UNKNOWN"))
        era_stats[era]["total"] += 1
        if r.get("era_correct"):
            era_stats[era]["correct"] += 1

    lines.append("| Era | Correct | Total | Accuracy |")
    lines.append("|-----|---------|-------|----------|")
    for era in sorted(era_stats.keys()):
        s = era_stats[era]
        pct = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        lines.append(f"| {era} | {s['correct']} | {s['total']} | {pct:.1f}% |")
    lines.append("")

    # Misrouted questions
    misrouted = [r for r in results if not r.get("era_correct")]
    if misrouted:
        lines.append("### Misrouted Questions")
        lines.append("")
        for r in misrouted:
            qid = r.get("id", "?")
            acceptable = r.get("acceptable_eras", [])
            bot_era = r.get("bot_era", "N/A")
            lines.append(f"- **{qid}**: expected {', '.join(acceptable)}, got {bot_era}")
        lines.append("")

    return "\n".join(lines)


def generate_full_report(data):
    """Generate a full evaluation report from scored results JSON."""
    meta = data["metadata"]
    results = data["results"]
    total = meta.get("total_questions", len(results))

    # Filter to scored results only
    scored = [r for r in results if r.get("status") == "scored" and r.get("scores")]
    scored_count = len(scored)

    lines = []
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append(f"**Date:** {meta.get('date', 'N/A')}")
    lines.append(f"**Responses file:** {meta.get('responses_file', 'N/A')}")
    lines.append("")

    # --- Summary Statistics ---
    lines.append("## Summary Statistics")
    lines.append("")

    if scored_count > 0:
        all_avgs = [avg_score(r) for r in scored if avg_score(r) is not None]
        overall_avg = mean(all_avgs) if all_avgs else 0
        pass_count = sum(1 for a in all_avgs if a >= PASS_THRESHOLD)
        pass_rate = (pass_count / len(all_avgs) * 100) if all_avgs else 0
    else:
        overall_avg = 0
        pass_count = 0
        pass_rate = 0

    era_correct_count = sum(1 for r in results if r.get("era_correct"))
    era_pct = (era_correct_count / total * 100) if total > 0 else 0
    misinfo_count = sum(1 for r in scored if r.get("scores", {}).get("no_misinformation") is False)

    lines.append(f"- **Total questions:** {total}")
    lines.append(f"- **Scored:** {scored_count}")
    lines.append(f"- **Overall average score:** {overall_avg:.2f}/5")
    lines.append(f"- **Pass rate (>= {PASS_THRESHOLD}):** {pass_count}/{scored_count} ({pass_rate:.1f}%)")
    lines.append(f"- **Era detection accuracy:** {era_correct_count}/{total} ({era_pct:.1f}%)")
    lines.append(f"- **Misinformation flags:** {misinfo_count}")
    lines.append("")

    # --- Era Detection Accuracy ---
    lines.append("## Era Detection Accuracy")
    lines.append("")
    lines.append(f"**Overall:** {era_correct_count}/{total} correct ({era_pct:.1f}%)")
    lines.append("")

    era_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        era = r.get("era", "UNKNOWN")
        era_stats[era]["total"] += 1
        if r.get("era_correct"):
            era_stats[era]["correct"] += 1

    lines.append("| Era | Correct | Total | Accuracy |")
    lines.append("|-----|---------|-------|----------|")
    for era in sorted(era_stats.keys()):
        s = era_stats[era]
        pct = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        lines.append(f"| {era} | {s['correct']} | {s['total']} | {pct:.1f}% |")
    lines.append("")

    misrouted = [r for r in results if not r.get("era_correct")]
    if misrouted:
        lines.append("### Misrouted Questions")
        lines.append("")
        for r in misrouted:
            qid = r.get("id", "?")
            question = r.get("question", "")[:80]
            bot_era = r.get("bot_era", "N/A")
            # Get acceptable_eras from golden test set or from result
            acceptable = r.get("acceptable_eras", [r.get("era", "?")])
            lines.append(f"- **{qid}**: \"{question}\" — expected {', '.join(acceptable)}, got {bot_era}")
        lines.append("")

    # --- Overall Scores ---
    lines.append("## Overall Scores")
    lines.append("")
    if scored_count > 0:
        for dim in SCORE_DIMS:
            dim_avg = mean(r["scores"][dim] for r in scored)
            lines.append(f"- **{dim}:** {dim_avg:.2f}/5")
    else:
        lines.append("No scored results available.")
    lines.append("")

    # --- Scores by Era ---
    lines.append("## Scores by Era")
    lines.append("")
    if scored_count > 0:
        era_scores = defaultdict(list)
        for r in scored:
            era_scores[r.get("era", "UNKNOWN")].append(r["scores"])

        lines.append("| Era | Factual | Era Align | Complete | Helpful | Avg |")
        lines.append("|-----|---------|-----------|----------|---------|-----|")
        for era in sorted(era_scores.keys()):
            scores_list = era_scores[era]
            fa = mean(s["factual_accuracy"] for s in scores_list)
            ea = mean(s["era_alignment"] for s in scores_list)
            co = mean(s["completeness"] for s in scores_list)
            he = mean(s["helpfulness"] for s in scores_list)
            avg = mean([fa, ea, co, he])
            lines.append(f"| {era} | {fa:.2f} | {ea:.2f} | {co:.2f} | {he:.2f} | {avg:.2f} |")
        lines.append("")

    # --- Scores by Category ---
    lines.append("## Scores by Category")
    lines.append("")
    if scored_count > 0:
        cat_scores = defaultdict(list)
        for r in scored:
            cat_scores[r.get("category", "unknown")].append(r["scores"])

        lines.append("| Category | Factual | Era Align | Complete | Helpful | Avg |")
        lines.append("|----------|---------|-----------|----------|---------|-----|")
        for cat in sorted(cat_scores.keys()):
            scores_list = cat_scores[cat]
            fa = mean(s["factual_accuracy"] for s in scores_list)
            ea = mean(s["era_alignment"] for s in scores_list)
            co = mean(s["completeness"] for s in scores_list)
            he = mean(s["helpfulness"] for s in scores_list)
            avg = mean([fa, ea, co, he])
            lines.append(f"| {cat} | {fa:.2f} | {ea:.2f} | {co:.2f} | {he:.2f} | {avg:.2f} |")
        lines.append("")

    # --- Bottom 10 Questions ---
    lines.append("## Bottom 10 Questions")
    lines.append("")
    if scored_count > 0:
        scored_with_avg = [(r, avg_score(r)) for r in scored if avg_score(r) is not None]
        scored_with_avg.sort(key=lambda x: x[1])
        bottom_10 = scored_with_avg[:10]

        for r, avg in bottom_10:
            qid = r.get("id", "?")
            question = r.get("question", "")
            response_excerpt = (r.get("bot_response") or "")[:200]
            scores = r["scores"]
            lines.append(f"### {qid} (avg: {avg:.2f})")
            lines.append(f"**Question:** {question}")
            lines.append(f"**Response excerpt:** {response_excerpt}...")
            lines.append(f"**Scores:** factual={scores['factual_accuracy']}, era_align={scores['era_alignment']}, complete={scores['completeness']}, helpful={scores['helpfulness']}")
            lines.append(f"**Misinformation:** {'No' if scores.get('no_misinformation', True) else 'YES'}")
            lines.append("")
    else:
        lines.append("No scored results available.")
        lines.append("")

    # --- Misinformation Flags ---
    lines.append("## Misinformation Flags")
    lines.append("")
    misinfo_results = [r for r in scored if r.get("scores", {}).get("no_misinformation") is False]
    if misinfo_results:
        for r in misinfo_results:
            qid = r.get("id", "?")
            question = r.get("question", "")
            bot_response = (r.get("bot_response") or "")[:300]
            lines.append(f"### {qid}")
            lines.append(f"**Question:** {question}")
            lines.append(f"**Bot response:** {bot_response}...")
            lines.append("")
    else:
        lines.append("No misinformation flags detected.")
        lines.append("")

    return "\n".join(lines)


def main(scored_file):
    """Generate report and return the output file path."""
    data = load_data(scored_file)

    if is_era_check_file(data):
        report_content = generate_era_only_report(data)
        prefix = "era_report"
    else:
        report_content = generate_full_report(data)
        prefix = "report"

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = os.path.join(RESULTS_DIR, f"{prefix}_{timestamp}.md")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)

    # Print summary to stdout
    meta = data.get("metadata", {})
    results = data.get("results", [])
    total = meta.get("total_questions", len(results))
    era_correct = sum(1 for r in results if r.get("era_correct"))
    era_pct = (era_correct / total * 100) if total > 0 else 0

    print(f"Era detection: {era_correct}/{total} correct ({era_pct:.1f}%)")

    if not is_era_check_file(data):
        scored = [r for r in results if r.get("status") == "scored" and r.get("scores")]
        if scored:
            all_avgs = [avg_score(r) for r in scored if avg_score(r) is not None]
            overall_avg = mean(all_avgs) if all_avgs else 0
            pass_count = sum(1 for a in all_avgs if a >= PASS_THRESHOLD)
            misinfo = sum(1 for r in scored if r.get("scores", {}).get("no_misinformation") is False)
            print(f"Overall average: {overall_avg:.2f}/5")
            print(f"Pass rate: {pass_count}/{len(scored)} ({(pass_count / len(scored) * 100):.1f}%)")
            print(f"Misinformation flags: {misinfo}")

    print(f"Report saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--scored-file", required=True,
                        help="Path to scored results JSON or era_check JSON")
    args = parser.parse_args()
    main(scored_file=args.scored_file)
