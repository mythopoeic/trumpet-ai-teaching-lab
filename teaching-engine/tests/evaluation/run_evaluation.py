"""Run golden test questions against the bot's /chat endpoint and capture responses.

Run from teaching-engine/ directory:
    python tests/evaluation/run_evaluation.py
    python tests/evaluation/run_evaluation.py --base-url http://localhost:8000
    python tests/evaluation/run_evaluation.py --quick
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Production golden set (excluded from the public portfolio snapshot).
GOLDEN_TEST_FILE = os.path.join(SCRIPT_DIR, "golden_test_set.json")
# Synthetic fallback shipped in the public snapshot (no private content).
_REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
SAMPLE_EVAL_FILE = os.path.join(
    _REPO_ROOT, "sample_data", "eval", "example_eval_set.json"
)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_test_set():
    """Load the evaluation question set.

    Prefers the production golden set when present; in the public portfolio
    snapshot that file is intentionally excluded, so this falls back to the
    synthetic sample set under sample_data/eval/. If neither exists, exits
    gracefully with a clear message rather than raising.
    """
    if os.path.exists(GOLDEN_TEST_FILE):
        with open(GOLDEN_TEST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    if os.path.exists(SAMPLE_EVAL_FILE):
        print(
            "Production golden evaluation set is excluded from this public "
            "portfolio snapshot; using the synthetic sample set "
            "(sample_data/eval/example_eval_set.json)."
        )
        with open(SAMPLE_EVAL_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(
        "No evaluation set found. The production golden set is excluded from "
        "this public portfolio snapshot and no synthetic sample set is present. "
        "See docs/portfolio-snapshot.md."
    )
    sys.exit(0)


def send_question(base_url, question_text):
    """Send a question to the /chat endpoint and return the response."""
    url = f"{base_url}/chat"
    payload = json.dumps({
        "text": question_text,
        "session_id": None,
        "era": None,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(base_url="http://localhost:8000", quick=False):
    """Run evaluation and return the output file path."""
    test_data = load_test_set()
    questions = test_data["questions"]
    total = len(questions)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    era_correct_count = 0

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question_text = q["question"]
        expected_era = q["era"]
        acceptable_eras = q.get("acceptable_eras", [expected_era])

        try:
            resp = send_question(base_url, question_text)
            bot_response = resp.get("answer", "")
            bot_era = resp.get("era", "")
            bot_citations = resp.get("citations", [])
            era_correct = bot_era in acceptable_eras
            status = "ok"
            error = None
        except urllib.error.HTTPError as e:
            bot_response = None
            bot_era = None
            bot_citations = []
            era_correct = False
            status = "error"
            error = f"HTTP {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            bot_response = None
            bot_era = None
            bot_citations = []
            era_correct = False
            status = "error"
            error = f"Connection error: {e.reason}"
        except Exception as e:
            bot_response = None
            bot_era = None
            bot_citations = []
            era_correct = False
            status = "error"
            error = str(e)

        if era_correct:
            era_correct_count += 1

        # Print progress
        era_check = "?" if bot_era is None else bot_era
        if status == "error":
            print(f"Q{i:03d} [{qid}]: ERROR - {error}")
        elif era_correct:
            print(f"Q{i:03d} [{qid}]: OK (era: {era_check} \u2713)")
        else:
            print(f"Q{i:03d} [{qid}]: OK (era: {era_check} \u2717 expected {','.join(acceptable_eras)})")

        result = {
            "id": qid,
            "question": question_text,
            "era": expected_era,
            "acceptable_eras": acceptable_eras,
            "category": q.get("category", ""),
            "bot_response": bot_response,
            "bot_era": bot_era,
            "era_correct": era_correct,
            "bot_citations": bot_citations,
            "status": status,
            "error": error,
        }
        results.append(result)

        # Small delay between requests
        if i < total:
            time.sleep(0.5)

    # Print summary
    pct = (era_correct_count / total * 100) if total > 0 else 0
    print(f"\nEra detection: {era_correct_count}/{total} correct ({pct:.1f}%)")

    # Save results
    if quick:
        # Lightweight era check file
        era_results = [{
            "id": r["id"],
            "expected_era": r["era"],
            "acceptable_eras": r["acceptable_eras"],
            "bot_era": r["bot_era"],
            "era_correct": r["era_correct"],
        } for r in results]
        output = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "base_url": base_url,
                "total_questions": total,
                "era_correct": era_correct_count,
                "era_accuracy_pct": round(pct, 1),
                "mode": "quick",
            },
            "results": era_results,
        }
        output_file = os.path.join(RESULTS_DIR, f"era_check_{timestamp}.json")
    else:
        output = {
            "metadata": {
                "date": datetime.now().isoformat(),
                "base_url": base_url,
                "total_questions": total,
                "era_correct": era_correct_count,
                "era_accuracy_pct": round(pct, 1),
            },
            "results": results,
        }
        output_file = os.path.join(RESULTS_DIR, f"responses_{timestamp}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run golden test evaluation")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the bot API (default: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only check era detection, save lightweight results")
    args = parser.parse_args()
    main(base_url=args.base_url, quick=args.quick)
