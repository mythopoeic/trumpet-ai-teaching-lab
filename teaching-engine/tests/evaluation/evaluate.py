"""Run the full evaluation pipeline with a single command.

Run from teaching-engine/ directory:
    python tests/evaluation/evaluate.py                    # Full pipeline
    python tests/evaluation/evaluate.py --quick            # Era detection only (fast, free)
    python tests/evaluation/evaluate.py --skip-scoring     # Skip LLM scoring
    python tests/evaluation/evaluate.py --base-url http://localhost:9000
"""

import argparse
import sys

from tests.evaluation.run_evaluation import main as run_evaluation
from tests.evaluation.score_responses import main as score_responses
from tests.evaluation.generate_report import main as generate_report


def main():
    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="Base URL of the bot API (default: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only check era detection, skip LLM scoring (fast and free)")
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Run test questions but skip LLM scoring, generate era-only report")
    args = parser.parse_args()

    quick = args.quick
    skip_scoring = args.skip_scoring or quick

    if quick:
        total_steps = 2
    elif skip_scoring:
        total_steps = 2
    else:
        total_steps = 3

    # Step 1: Run test questions
    step = 1
    if quick:
        print(f"Step {step}/{total_steps}: Running era detection check...")
    else:
        print(f"Step {step}/{total_steps}: Running test questions...")
    print()

    try:
        responses_file = run_evaluation(base_url=args.base_url, quick=quick)
    except Exception as e:
        print(f"\nERROR in step {step}: {e}", file=sys.stderr)
        sys.exit(1)

    if not responses_file:
        print("\nERROR: Test runner did not produce an output file.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Score responses (unless skipping)
    scored_file = None
    if not skip_scoring:
        step = 2
        print(f"\nStep {step}/{total_steps}: Scoring responses with Claude...")
        print()

        try:
            scored_file = score_responses(responses_file=responses_file)
        except Exception as e:
            print(f"\nERROR in step {step}: {e}", file=sys.stderr)
            sys.exit(1)

        if not scored_file:
            print("\nERROR: Scorer did not produce an output file.", file=sys.stderr)
            sys.exit(1)

    # Step 3 (or 2): Generate report
    step = total_steps
    print(f"\nStep {step}/{total_steps}: Generating report...")
    print()

    report_input = scored_file if scored_file else responses_file
    try:
        report_file = generate_report(scored_file=report_input)
    except Exception as e:
        print(f"\nERROR in step {step}: {e}", file=sys.stderr)
        sys.exit(1)

    if not report_file:
        print("\nERROR: Report generator did not produce an output file.", file=sys.stderr)
        sys.exit(1)

    print(f"\nEvaluation complete! Full report: {report_file}")


if __name__ == "__main__":
    main()
