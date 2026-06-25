"""
src/evals/run_evals.py
-----------------------
LangSmith evaluation suite for the RAG pipeline.

This is the most important file for an AI-FDE portfolio.
It proves you can:
1. Define measurable quality criteria
2. Run automated evals on every change
3. Detect regressions before they hit production
4. Present results to non-technical stakeholders

Run with: python src/evals/run_evals.py
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from typing import Any

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from src.rag.pipeline import EnterpriseRAGPipeline
from src.evals.datasets import get_or_create_eval_dataset
from src.utils.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── LangSmith client ──────────────────────────────────────────────────────────

ls_client = Client()


# ── Evaluators ────────────────────────────────────────────────────────────────

def faithfulness_evaluator(run: Run, example: Example) -> dict:
    """
    Custom faithfulness evaluator.
    Checks if the answer is supported by the retrieved context.

    This is the #1 enterprise concern: hallucination.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")

    if not answer or not context:
        return {"key": "faithfulness", "score": 0.0, "comment": "Missing answer or context"}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rate the faithfulness of this answer on a scale of 0.0 to 1.0.

1.0 = Every claim is directly supported by the context
0.5 = Some claims supported, some unsupported
0.0 = Answer contradicts or ignores the context (hallucination)

Respond ONLY with valid JSON: {{"score": 0.0, "reason": "brief explanation"}}"""),
        ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "answer": answer})

    try:
        parsed = json.loads(result)
        score = float(parsed.get("score", 0.0))
        comment = parsed.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        comment = "Parse error in evaluator"

    return {"key": "faithfulness", "score": score, "comment": comment}


def answer_relevance_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluates if the answer actually addresses the question.
    High faithfulness + low relevance = retrieved wrong docs.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    question = example.inputs.get("query", "")
    answer = run.outputs.get("answer", "")

    if not question or not answer:
        return {"key": "answer_relevance", "score": 0.0}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rate how well the answer addresses the question on a scale of 0.0 to 1.0.

1.0 = Fully and directly answers the question
0.5 = Partially answers the question
0.0 = Does not address the question at all

Respond ONLY with valid JSON: {{"score": 0.0, "reason": "brief explanation"}}"""),
        ("human", "Question: {question}\n\nAnswer: {answer}")
    ])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "answer": answer})

    try:
        parsed = json.loads(result)
        score = float(parsed.get("score", 0.0))
        comment = parsed.get("reason", "")
    except (json.JSONDecodeError, ValueError):
        score = 0.5
        comment = "Parse error"

    return {"key": "answer_relevance", "score": score, "comment": comment}


def no_hallucination_evaluator(run: Run, example: Example) -> dict:
    """
    Binary evaluator: did the model hallucinate?
    Used for CI/CD — if hallucination detected, fail the build.
    """
    needs_review = run.outputs.get("needs_human_review", False)
    confidence = run.outputs.get("confidence", 1.0)

    # Flagged for review OR very low confidence = potential hallucination
    is_clean = not needs_review and confidence > 0.6
    score = 1.0 if is_clean else 0.0

    return {
        "key": "no_hallucination",
        "score": score,
        "comment": f"confidence={confidence:.2f}, needs_review={needs_review}"
    }


def latency_evaluator(run: Run, example: Example) -> dict:
    """
    Latency eval — production systems must meet SLA.
    Target: < 3 seconds per query.
    """
    if run.end_time and run.start_time:
        latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
        # Score: 1.0 if under 3s, decreasing linearly to 0 at 10s
        score = max(0.0, min(1.0, 1.0 - (latency_ms - 3000) / 7000))
        return {
            "key": "latency_ok",
            "score": score,
            "comment": f"{latency_ms:.0f}ms"
        }
    return {"key": "latency_ok", "score": 0.5, "comment": "Could not measure latency"}


# ── Pipeline wrapper for LangSmith evaluate() ────────────────────────────────

def _build_pipeline_runner(pipeline: EnterpriseRAGPipeline):
    """Return a closure that runs the pipeline — defers construction to run_evaluation()."""
    def run_pipeline(inputs: dict[str, Any]) -> dict[str, Any]:
        """Wrapper that maps LangSmith example inputs → pipeline outputs."""
        query = inputs.get("query", "")
        response = pipeline.invoke(query)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence,
            "needs_human_review": response.needs_human_review,
            "query_rewritten": response.query_rewritten,
            # context captured inline; in production pull from LangSmith trace
            "context": inputs.get("_context_for_eval", ""),
        }
    return run_pipeline


# ── Main eval runner ──────────────────────────────────────────────────────────

def run_evaluation(fail_below: float = 0.75) -> dict:
    """
    Run the full eval suite against the LangSmith dataset.

    Args:
        fail_below: If any metric avg drops below this, exit with error code 1.
                    Used in CI/CD to catch regressions.

    Returns:
        dict with all metric scores
    """
    logger.info("=" * 60)
    logger.info("Starting RAG Evaluation Suite")
    logger.info("Project: %s", settings.LANGCHAIN_PROJECT)
    logger.info("=" * 60)

    # Build pipeline here so importing this module has no side-effects
    pipeline = EnterpriseRAGPipeline()
    run_pipeline = _build_pipeline_runner(pipeline)

    # Get or create the eval dataset in LangSmith
    dataset_name = "enterprise-rag-eval-v1"
    dataset = get_or_create_eval_dataset(ls_client, dataset_name)
    logger.info("Using dataset: %s (%s examples)", dataset_name, dataset.example_count)

    experiment_prefix = f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M')}"

    results = evaluate(
        run_pipeline,
        data=dataset_name,
        evaluators=[
            faithfulness_evaluator,
            answer_relevance_evaluator,
            no_hallucination_evaluator,
            latency_evaluator,
        ],
        experiment_prefix=experiment_prefix,
        metadata={
            "model": settings.OPENAI_MODEL,
            "pipeline_version": "1.0.0",
            "chunking_strategy": "recursive-512-64",
            "retrieval_mode": "hybrid",
        },
        max_concurrency=3,
    )

    # Aggregate scores
    scores: dict[str, list[float]] = {
        "faithfulness": [],
        "answer_relevance": [],
        "no_hallucination": [],
        "latency_ok": [],
    }

    for result in results:
        for eval_result in result.get("evaluation_results", {}).get("results", []):
            key = eval_result.key
            if key in scores and eval_result.score is not None:
                scores[key].append(eval_result.score)

    averages = {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}

    # Print results table
    logger.info("\n%s", "=" * 60)
    logger.info("EVAL RESULTS")
    logger.info("%s", "=" * 60)
    logger.info("%-25s %-10s %-10s %s", "Metric", "Score", "Threshold", "Status")
    logger.info("%s", "-" * 60)

    thresholds = {
        "faithfulness": 0.85,
        "answer_relevance": 0.80,
        "no_hallucination": 0.90,
        "latency_ok": 0.75,
    }

    all_pass = True
    for metric, avg_score in averages.items():
        threshold = thresholds.get(metric, fail_below)
        passed = avg_score >= threshold
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        logger.info("%-25s %-10.3f %-10.3f %s", metric, avg_score, threshold, status)

    logger.info("%s", "=" * 60)
    logger.info(
        "Overall: %s | LangSmith: https://smith.langchain.com",
        "ALL PASS" if all_pass else "REGRESSIONS DETECTED"
    )
    logger.info("%s", "=" * 60)

    if not all_pass:
        logger.error("Eval suite FAILED — regressions detected above threshold")
        sys.exit(1)

    return averages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument(
        "--fail-below",
        type=float,
        default=0.75,
        help="Exit with error if any metric falls below this score (default: 0.75)"
    )
    args = parser.parse_args()
    run_evaluation(fail_below=args.fail_below)
