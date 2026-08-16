from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    HF_TOKEN: str = ""
    CONFIDENCE_THRESHOLD: float = 0.75
    MODEL: str = "meta-llama/Llama-3.1-8B-Instruct"
    API_URL: str = "https://router.huggingface.co/v1/chat/completions"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
