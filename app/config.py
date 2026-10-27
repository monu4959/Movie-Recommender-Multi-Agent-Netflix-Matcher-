from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "movies"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    openai_chat_model: str = "gpt-4o"
    openai_embed_model: str = "text-embedding-3-small"
    embed_dim: int = 1536

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()