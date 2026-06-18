"""Score bot responses against golden test reference answers using Claude as judge.

Run from teaching-engine/ directory:
    python tests/evaluation/score_responses.py --responses-file results/responses_2026-02-16_120000.json
"""

import argparse
import json
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_TEST_FILE = os.path.join(SCRIPT_DIR, "golden_test_set.json")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

SCORING_SYSTEM_PROMPT = """\
You are an evaluation judge for a trumpet teaching AI that teaches Jerome Callet's methods.

Score the bot's response against the reference answer and key facts. Return ONLY a JSON object with these keys:

- factual_accuracy (1-5): How factually correct is the response compared to the reference?
  5=all facts correct, 4=mostly correct with minor omissions, 3=some correct but missing key facts, 2=significant errors, 1=mostly incorrect
- era_alignment (1-5): Does the response use concepts and terminology appropriate to the expected era?
  5=perfectly aligned, 4=mostly aligned, 3=partially aligned, 2=wrong era concepts, 1=completely wrong era
- completeness (1-5): How well does the response cover the key facts?
  5=covers all key facts, 4=covers most, 3=covers some, 2=covers few, 1=covers none
- helpfulness (1-5): How helpful would this response be to a trumpet student?
  5=excellent teaching, 4=good guidance, 3=adequate, 2=confusing or vague, 1=unhelpful or misleading
- no_misinformation (boolean): true if the response does NOT contain information that contradicts the reference material. false if it contains contradictory or fabricated claims.

Return ONLY the JSON object, no other text."""

SCORING_USER_TEMPLATE = """\
Question: {question}

Expected Era: {era}

Reference Answer: {reference_answer}

Key Facts:
{key_facts}

Bot's Response:
{bot_response}"""


def load_golden_test_set():
    with open(GOLDEN_TEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_responses(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_lookup(test_data):
    """Build a dict of question ID -> question data for quick lookup."""
    return {q["id"]: q for q in test_data["questions"]}


def score_one(client, question_data, bot_response_text):
    """Score a single bot response using Claude as judge."""
    key_facts_str = "\n".join(f"- {fact}" for fact in question_data["key_facts"])

    user_message = SCORING_USER_TEMPLATE.format(
        question=question_data["question"],
        era=question_data["era"],
        reference_answer=question_data["reference_answer"],
        key_facts=key_facts_str,
        bot_response=bot_response_text,
    )

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        temperature=0.0,
        system=SCORING_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    response_text = response.content[0].text
    scores = json.loads(response_text)
    return scores


def main(responses_file):
    """Score responses and return the output file path."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        return None

    client = Anthropic(api_key=api_key)

    # Load data
    test_data = load_golden_test_set()
    question_lookup = build_question_lookup(test_data)

    responses_data = load_responses(responses_file)
    response_results = responses_data["results"]
    total = len(response_results)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scored_results = []
    scored_count = 0
    skipped_count = 0

    for i, result in enumerate(response_results, 1):
        qid = result["id"]
        bot_response = result.get("bot_response")
        status = result.get("status", "ok")

        # Skip questions where bot_response is null or status is 'error'
        if bot_response is None or status == "error":
            print(f"  [{i:03d}/{total}] {qid}: SKIPPED (no response)")
            skipped_count += 1
            scored_entry = {
                "id": qid,
                "question": result.get("question", ""),
                "era": result.get("era", ""),
                "category": result.get("category", ""),
                "bot_response": bot_response,
                "bot_era": result.get("bot_era"),
                "era_correct": result.get("era_correct", False),
                "bot_citations": result.get("bot_citations", []),
                "status": "skipped",
                "scores": None,
            }
            scored_results.append(scored_entry)
            continue

        # Look up reference data
        question_data = question_lookup.get(qid)
        if not question_data:
            print(f"  [{i:03d}/{total}] {qid}: SKIPPED (not in golden test set)")
            skipped_count += 1
            continue

        # Score with Claude
        try:
            scores = score_one(client, question_data, bot_response)
            scored_count += 1
            mean_score = sum(scores[k] for k in ["factual_accuracy", "era_alignment", "completeness", "helpfulness"]) / 4
            print(f"  [{i:03d}/{total}] {qid}: scored (avg {mean_score:.1f}/5)")
        except json.JSONDecodeError as e:
            print(f"  [{i:03d}/{total}] {qid}: SCORE PARSE ERROR - {e}")
            scores = None
        except Exception as e:
            print(f"  [{i:03d}/{total}] {qid}: SCORING ERROR - {e}")
            scores = None

        scored_entry = {
            "id": qid,
            "question": result.get("question", ""),
            "era": result.get("era", ""),
            "category": result.get("category", ""),
            "bot_response": bot_response,
            "bot_era": result.get("bot_era"),
            "era_correct": result.get("era_correct", False),
            "bot_citations": result.get("bot_citations", []),
            "status": "scored" if scores else "score_error",
            "scores": scores,
        }
        scored_results.append(scored_entry)

        # Small delay between API calls
        if i < total:
            time.sleep(0.5)

    # Print summary
    print(f"\nScoring complete: {scored_count} scored, {skipped_count} skipped")

    # Save scored results
    output = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "responses_file": responses_file,
            "total_questions": total,
            "scored": scored_count,
            "skipped": skipped_count,
        },
        "results": scored_results,
    }

    output_file = os.path.join(RESULTS_DIR, f"scored_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Scored results saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score bot responses using Claude as judge")
    parser.add_argument("--responses-file", required=True,
                        help="Path to responses JSON file from run_evaluation.py")
    args = parser.parse_args()
    main(responses_file=args.responses_file)
