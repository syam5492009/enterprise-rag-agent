"""
src/utils/config.py — Settings management via pydantic-settings
"""
from dotenv import load_dotenv
load_dotenv()  # put .env vars into os.environ so LangChain picks them up

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # LangSmith — set these and every LLM call is automatically traced
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "enterprise-rag-agent"

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_PATH: str = ""  # local file path — set this to run without Docker
    QDRANT_COLLECTION: str = "enterprise_kb"

    # Optional: Confluence integration
    CONFLUENCE_URL: str = ""
    CONFLUENCE_USERNAME: str = ""
    CONFLUENCE_API_KEY: str = ""


settings = Settings()
