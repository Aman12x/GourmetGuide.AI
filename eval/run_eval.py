"""Live evaluation of the GourmetGuide pipeline on Amazon Bedrock.

    python eval/run_eval.py                       # all cases, writes eval/results/<timestamp>.json
    python eval/run_eval.py --limit 5             # quick smoke run

Per case it checks intent and filter extraction against the labels, whether any labelled dish
is recommended (hit rate), whether any recommendation breaks the request's hard constraints
(must be zero), and asks a judge model on Bedrock whether each summary is grounded in the dish
data. Exits 1 when a gate fails. Needs AWS credentials with Bedrock access.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bedrock import DEFAULT_MODEL_ID, EMBEDDING_MODEL_ID, BedrockLLM  # noqa: E402
from pipeline import plan_turn, rank_and_summarize  # noqa: E402
from retrieval import Filters, allergen_terms, matches, search  # noqa: E402

CASES = ROOT / "eval/cases.jsonl"
JUDGE_MODEL_ID = "us.anthropic.claude-sonnet-5-5"
GATES = {"intent_accuracy": 0.9, "filter_accuracy": 0.85, "hit_rate": 0.8, "groundedness": 0.9, "constraint_violations": 0}

JUDGE_SYSTEM = (
    "You check a restaurant recommendation for hallucination. Given a dish's source data and a summary "
    "written about it, decide whether every factual claim in the summary (ingredients, price, cuisine, "
    "restaurant, dietary facts) is supported by the source data. Style and enthusiasm are fine."
)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}, "unsupported_claims": {"type": "array", "items": {"type": "string"}}},
    "required": ["grounded", "unsupported_claims"],
}


def load_cases(path=CASES):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def same_allergens(got, want):
    norm = lambda xs: {tuple(allergen_terms(x)) for x in xs}  # noqa: E731
    return norm(got) >= norm(want)


def field_ok(key, got, want):
    if key == "cuisines":
        return {c.lower() for c in want} <= {c.lower() for c in (got or [])}
    if key == "exclude_allergens":
        return same_allergens(got or [], want)
    if key in ("max_price", "min_rating"):
        return got is not None and abs(float(got) - float(want)) < 1e-6
    return got == want


def judge(judge_llm, metadata, page_content, summary):
    source = json.dumps({k: v for k, v in metadata.items() if k != "image_path"}) + "\n" + page_content
    message = {"role": "user", "content": [{"text": f"Source data:\n{source}\n\nSummary:\n{summary}"}]}
    return judge_llm.structured(JUDGE_SYSTEM, [message], "verdict", "Return the grounding verdict.", JUDGE_SCHEMA)


def run_case(case, llm, judge_llm, db, cuisines):
    started, calls_before = time.perf_counter(), llm.calls
    plan = plan_turn(llm, case["query"], "", [], cuisines)
    want = dict(case["expect"])
    want_intent = want.pop("intent", "recommend")
    result = {"id": case["id"], "query": case["query"], "plan": plan, "intent_ok": plan.get("intent") == want_intent}
    result["filters_ok"] = {k: field_ok(k, plan.get(k), v) for k, v in want.items()}

    picks = []
    if plan.get("intent") == "recommend" and want_intent == "recommend":
        filters = Filters.from_plan(plan)
        docs = search(db, plan["search_query"], filters)
        result["retrieved"] = [d.metadata["id"] for d in docs]
        if docs:
            _, picks = rank_and_summarize(llm, docs, case["query"])
        pages = {d.metadata["id"]: d.page_content for d in docs}
        expected_filters = Filters.from_plan(want)
        result["picks"] = []
        for metadata, summary in picks:
            verdict = judge(judge_llm, metadata, pages[metadata["id"]], summary)
            result["picks"].append(
                {"id": metadata["id"], "summary": summary, "violates": not matches(metadata, expected_filters), **verdict}
            )
        result["hit"] = any(p["id"] in case["relevant"] for p in result["picks"])
    result["model_calls"] = llm.calls - calls_before
    result["seconds"] = round(time.perf_counter() - started, 2)
    return result


def summarize(results):
    rec = [r for r in results if "hit" in r]
    filter_checks = [ok for r in results for ok in r["filters_ok"].values()]
    picks = [p for r in rec for p in r["picks"]]
    return {
        "cases": len(results),
        "intent_accuracy": sum(r["intent_ok"] for r in results) / len(results),
        "filter_accuracy": sum(filter_checks) / len(filter_checks) if filter_checks else 1.0,
        "hit_rate": sum(r["hit"] for r in rec) / len(rec) if rec else 0.0,
        "groundedness": sum(p["grounded"] for p in picks) / len(picks) if picks else 0.0,
        "constraint_violations": sum(p["violates"] for p in picks),
        "mean_model_calls": sum(r["model_calls"] for r in results) / len(results),
        "mean_seconds": sum(r["seconds"] for r in results) / len(results),
    }


def gate(summary):
    failures = [k for k, floor in GATES.items() if k != "constraint_violations" and summary[k] < floor]
    if summary["constraint_violations"] > GATES["constraint_violations"]:
        failures.append("constraint_violations")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--judge-model", default=JUDGE_MODEL_ID)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    import boto3
    from langchain_aws import BedrockEmbeddings
    from langchain_community.vectorstores import FAISS

    client = boto3.client("bedrock-runtime", region_name=args.region)
    db = FAISS.load_local(
        str(ROOT / "output/faiss_index"),
        BedrockEmbeddings(client=client, model_id=EMBEDDING_MODEL_ID),
        allow_dangerous_deserialization=True,  # built locally by ingest.py
    )
    cuisines = sorted({d.metadata["cuisine"] for d in db.docstore._dict.values()})
    llm, judge_llm = BedrockLLM(client, args.model), BedrockLLM(client, args.judge_model)

    cases = load_cases()[: args.limit]
    results = [run_case(c, llm, judge_llm, db, cuisines) for c in cases]
    summary = summarize(results)
    failures = gate(summary)

    out_dir = ROOT / "eval/results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"{stamp}.json"
    out.write_text(json.dumps({"model": args.model, "judge": args.judge_model, "summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"results: {out}")
    if failures:
        print("GATE FAILED: " + ", ".join(failures))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
