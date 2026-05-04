"""FinGent settings.

Pydantic Settings reads `.env` from the current working directory. The
runtime defaults are reasonable for the cheapest path (DeepSeek v4
Flash + Bedrock embeddings); override via .env or environment.

Note: KB_ID and KB_STORE_DIR are user-supplied at Agent / build_kb call
time, not env-driven, so the same FinGent install can address multiple
KBs.
"""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Sync .env into os.environ at import time. pydantic-settings already
# reads .env into Settings fields, but langchain-core's tracing layer
# reads LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT
# directly from os.environ, so without this propagation those vars
# stay invisible to langchain even when .env says LANGSMITH_TRACING
# is true. load_dotenv defaults to override=False so any value the
# user has already set in their shell wins over .env.
# Resolve relative to this file: src/fingent/config.py → ../../.env
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === Provider selection ===

    ORCHESTRATOR_PROVIDER: Literal["bedrock", "deepseek"] = Field(
        default="deepseek",
        description=(
            "Which LLM provider serves the orchestrator (the ReAct agent). "
            "Router and judge stay on Bedrock Haiku regardless."
        ),
    )

    # === Bedrock ===

    AWS_REGION: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock embeddings + (optional) Bedrock orchestrator.",
    )

    ORCHESTRATOR_MODEL_ID: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        description="Bedrock model ID for the main agent (used when provider=bedrock).",
    )

    ROUTER_MODEL_ID: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Bedrock model ID for intent classification.",
    )

    # === DeepSeek ===

    DEEPSEEK_API_KEY: str = Field(
        default="",
        description="API key for DeepSeek (OpenAI-compatible endpoint).",
    )

    DEEPSEEK_MODEL_ID: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek model ID (used when provider=deepseek).",
    )

    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek OpenAI-compatible API base URL.",
    )


settings = Settings()
