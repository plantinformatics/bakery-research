"""Run the Jev scope and Pretzel checks from `Query.py` over the retained
test questions and report each probability against its expected label.

    cd GraphRAG && .venv/bin/python evals/run_jev_gates.py [questions.json]

Uses the same question definitions, state and thresholds as the pipeline,
so wording or threshold changes in `Query.py` are measured directly.
`null` labels are borderline cases: shown, but not counted.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typesafe_sdk import TypeSafeClient  # noqa: E402

from Query import (  # noqa: E402
    JEV_ASSISTANT_SCOPE,
    JEV_IN_SCOPE_MIN_PROBABILITY,
    JEV_MODEL,
    JEV_PRETZEL_DESCRIPTION,
    JEV_PRETZEL_MIN_PROBABILITY,
    JEV_PRETZEL_QUESTIONS,
    JEV_SCOPE_QUESTIONS,
)

QUESTIONS_PATH = Path(__file__).resolve().parent / "jev_gate_questions.json"


def evaluate(client: TypeSafeClient, item: dict) -> dict:
    question = item["question"]
    scope = client.system_one(
        state={"question": question, "assistant_scope": JEV_ASSISTANT_SCOPE},
        questions=JEV_SCOPE_QUESTIONS,
    )
    pretzel = client.system_one(
        state={"question": question, "pretzel": JEV_PRETZEL_DESCRIPTION},
        questions=JEV_PRETZEL_QUESTIONS,
    )
    return {
        **item,
        "in_scope_probability": scope.answers["in_scope"].noul,
        "pretzel_probability": pretzel.answers["pretzel_how_to"].noul,
    }


def verdict(expected, probability: float, threshold: float) -> str:
    if expected is None:
        return "  "
    return "ok" if (probability >= threshold) == expected else "XX"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else QUESTIONS_PATH
    items = json.loads(path.read_text(encoding="utf-8"))["questions"]
    with TypeSafeClient(api_key=os.environ["JEV_API_KEY"], model=JEV_MODEL) as client:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda item: evaluate(client, item), items))

    misses = 0
    for result in results:
        scope_verdict = verdict(result["in_scope"], result["in_scope_probability"], JEV_IN_SCOPE_MIN_PROBABILITY)
        pretzel_verdict = verdict(result["pretzel_how_to"], result["pretzel_probability"], JEV_PRETZEL_MIN_PROBABILITY)
        misses += (scope_verdict == "XX") + (pretzel_verdict == "XX")
        print(
            f"{scope_verdict} scope={result['in_scope_probability']:.2f}  "
            f"{pretzel_verdict} pretzel={result['pretzel_probability']:.2f}  {result['question']}"
        )
    labelled = sum(r["in_scope"] is not None for r in results) + sum(
        r["pretzel_how_to"] is not None for r in results
    )
    print(
        f"\n{labelled - misses}/{labelled} labelled checks correct "
        f"(in-scope >= {JEV_IN_SCOPE_MIN_PROBABILITY}, Pretzel >= {JEV_PRETZEL_MIN_PROBABILITY})"
    )


if __name__ == "__main__":
    main()
