"""
src/utils/langsmith_utils.py
-----------------------------
LangSmith tracing utilities.
Import this in any module to add custom tags, metadata, or feedback.
"""

from langsmith import Client
from functools import wraps
import logging

logger = logging.getLogger(__name__)
_client = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def get_run_url(run_id: str) -> str:
    """Get the LangSmith UI URL for a specific run (useful for debugging)."""
    return f"https://smith.langchain.com/runs/{run_id}"


def tag_run(tags: list[str]):
    """
    Decorator to add tags to a LangSmith run.
    Use to mark prod vs staging, or by team/feature.
    
    Example:
        @tag_run(["production", "rag-pipeline"])
        def my_function(...):
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_user_feedback(run_id: str, score: float, comment: str = ""):
    """Log human feedback score to a LangSmith run."""
    try:
        client = get_client()
        client.create_feedback(
            run_id=run_id,
            key="user_rating",
            score=score,
            comment=comment,
        )
        logger.info("Feedback logged for run %s: %.2f", run_id, score)
    except Exception as e:
        logger.error("Failed to log feedback: %s", e)
