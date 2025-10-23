"""Settings for the project."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base directory for storing content
    content_base_dir: str = "content"

    # LLM Provider Settings for contextual chunking
    llm_skip_context: bool = False
    llm_context_model: str = "qwen2.5:7b-instruct"
    llm_openai_url: str | None = None
    llm_watsonx_url: str | None = None
    llm_context_limit_override: int | None = None

    # WatsonX credentials (only needed if using WatsonX provider)
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""

    # Chunking Settings
    chunking_pattern: str = "**/*.json"

    # Embedding Settings
    embedding_model: str = "granite-30m-english"
    embedding_pattern: str = "**/*.chunks.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
