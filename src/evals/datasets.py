"""
src/evals/datasets.py
----------------------
Manages the evaluation dataset in LangSmith.
Contains 25 enterprise QA examples covering:
- Factual retrieval questions
- Multi-document synthesis questions  
- Edge cases (unanswerable questions, ambiguous queries)
- Questions designed to trigger hallucination

A strong eval dataset is what proves your RAG system is production-ready.
"""

from langsmith import Client
from langsmith.schemas import Dataset
import logging

logger = logging.getLogger(__name__)

# ── Eval dataset ──────────────────────────────────────────────────────────────
# In a real deployment, these come from actual user queries + expert annotations.
# For the portfolio, these demonstrate coverage across query types.

EVAL_EXAMPLES = [
    # --- Factual retrieval ---
    {
        "inputs": {"query": "What is the company's data retention policy for customer PII?"},
        "outputs": {"expected_topics": ["retention", "PII", "policy", "data"]},
        "metadata": {"category": "factual", "difficulty": "easy"}
    },
    {
        "inputs": {"query": "How do I request access to the production database?"},
        "outputs": {"expected_topics": ["access request", "production", "database", "approval"]},
        "metadata": {"category": "factual", "difficulty": "easy"}
    },
    {
        "inputs": {"query": "What are the SLA commitments for the API gateway?"},
        "outputs": {"expected_topics": ["SLA", "uptime", "latency", "API gateway"]},
        "metadata": {"category": "factual", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "Which AWS regions are approved for storing healthcare data?"},
        "outputs": {"expected_topics": ["AWS", "regions", "HIPAA", "compliance", "healthcare"]},
        "metadata": {"category": "factual", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "What is the process for deploying hotfixes to production?"},
        "outputs": {"expected_topics": ["hotfix", "deployment", "approval", "rollback"]},
        "metadata": {"category": "factual", "difficulty": "medium"}
    },

    # --- Multi-document synthesis ---
    {
        "inputs": {"query": "How does our incident response process differ for P1 vs P2 incidents?"},
        "outputs": {"expected_topics": ["P1", "P2", "response time", "escalation", "on-call"]},
        "metadata": {"category": "synthesis", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "What security controls apply when integrating third-party APIs?"},
        "outputs": {"expected_topics": ["security", "API", "authentication", "audit", "review"]},
        "metadata": {"category": "synthesis", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "Compare the authentication options available for mobile vs web apps"},
        "outputs": {"expected_topics": ["OAuth", "SSO", "mobile", "web", "authentication"]},
        "metadata": {"category": "synthesis", "difficulty": "hard"}
    },

    # --- Ambiguous queries (tests query rewriting) ---
    {
        "inputs": {"query": "What's the limit?"},
        "outputs": {"expected_topics": ["clarification", "rate limit", "context needed"]},
        "metadata": {"category": "ambiguous", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "How do I set it up?"},
        "outputs": {"expected_topics": ["setup", "configuration"]},
        "metadata": {"category": "ambiguous", "difficulty": "hard"}
    },

    # --- Unanswerable questions (tests "I don't know" behavior) ---
    {
        "inputs": {"query": "What will the company's revenue be next quarter?"},
        "outputs": {"expected_behavior": "should_decline", "expected_topics": ["insufficient information"]},
        "metadata": {"category": "unanswerable", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "Who is the CEO's personal phone number?"},
        "outputs": {"expected_behavior": "should_decline"},
        "metadata": {"category": "unanswerable", "difficulty": "easy"}
    },

    # --- Hallucination-prone queries ---
    {
        "inputs": {"query": "List all 47 microservices in our architecture"},
        "outputs": {"expected_topics": ["microservices", "architecture"]},
        "metadata": {"category": "hallucination_risk", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "What exact cost savings did the migration produce?"},
        "outputs": {"expected_topics": ["cost", "migration", "savings"]},
        "metadata": {"category": "hallucination_risk", "difficulty": "hard"}
    },

    # --- Technical operations ---
    {
        "inputs": {"query": "How do I rotate the JWT signing keys without downtime?"},
        "outputs": {"expected_topics": ["JWT", "key rotation", "zero downtime", "authentication"]},
        "metadata": {"category": "technical", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "What monitoring alerts are configured for the payment service?"},
        "outputs": {"expected_topics": ["monitoring", "alerts", "payment", "Datadog", "PagerDuty"]},
        "metadata": {"category": "technical", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "How do I enable debug logging in production without redeploying?"},
        "outputs": {"expected_topics": ["logging", "debug", "feature flags", "environment variables"]},
        "metadata": {"category": "technical", "difficulty": "medium"}
    },

    # --- Compliance and policy ---
    {
        "inputs": {"query": "What GDPR obligations apply when processing EU customer data?"},
        "outputs": {"expected_topics": ["GDPR", "EU", "data processing", "consent", "DPA"]},
        "metadata": {"category": "compliance", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "Is it permitted to store API keys in environment variables?"},
        "outputs": {"expected_topics": ["API keys", "secrets management", "Vault", "security"]},
        "metadata": {"category": "compliance", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "What is the approval process for open-source library usage?"},
        "outputs": {"expected_topics": ["open source", "license", "approval", "security scan"]},
        "metadata": {"category": "compliance", "difficulty": "medium"}
    },

    # --- Onboarding and HR ---
    {
        "inputs": {"query": "What access should a new backend engineer have on day one?"},
        "outputs": {"expected_topics": ["onboarding", "access", "permissions", "dev environment"]},
        "metadata": {"category": "onboarding", "difficulty": "easy"}
    },
    {
        "inputs": {"query": "How do I submit a request for a new software license?"},
        "outputs": {"expected_topics": ["license", "software", "procurement", "approval"]},
        "metadata": {"category": "onboarding", "difficulty": "easy"}
    },

    # --- Cross-system queries (tests MCP integration) ---
    {
        "inputs": {"query": "What JIRA tickets are blocking the API v3 release?"},
        "outputs": {"expected_topics": ["JIRA", "blockers", "API", "release"]},
        "metadata": {"category": "cross_system", "difficulty": "hard"}
    },
    {
        "inputs": {"query": "Show me the Confluence page for the database schema changelog"},
        "outputs": {"expected_topics": ["Confluence", "schema", "changelog", "database"]},
        "metadata": {"category": "cross_system", "difficulty": "medium"}
    },
    {
        "inputs": {"query": "What were the action items from last week's architecture review?"},
        "outputs": {"expected_topics": ["architecture review", "action items", "meeting"]},
        "metadata": {"category": "cross_system", "difficulty": "medium"}
    },
]


def get_or_create_eval_dataset(client: Client, dataset_name: str) -> Dataset:
    """
    Get existing eval dataset or create it from the examples above.
    
    In production, you'd grow this dataset from:
    1. Real user queries marked as good/bad by the team
    2. Edge cases discovered in production traces
    3. Adversarial examples from red-teaming sessions
    """
    # Check if dataset already exists
    existing = list(client.list_datasets(dataset_name=dataset_name))
    if existing:
        logger.info("Using existing eval dataset: %s", dataset_name)
        return existing[0]

    # Create new dataset
    logger.info("Creating eval dataset: %s with %d examples", dataset_name, len(EVAL_EXAMPLES))
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=(
            "Enterprise RAG evaluation dataset. "
            "Covers factual retrieval, synthesis, ambiguous queries, "
            "unanswerable questions, and hallucination-prone scenarios."
        ),
    )

    # Upload examples
    client.create_examples(
        inputs=[e["inputs"] for e in EVAL_EXAMPLES],
        outputs=[e.get("outputs", {}) for e in EVAL_EXAMPLES],
        metadata=[e.get("metadata", {}) for e in EVAL_EXAMPLES],
        dataset_id=dataset.id,
    )

    logger.info("Dataset created with %d examples", len(EVAL_EXAMPLES))
    return dataset
