from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_ignore_empty: blank placeholders in .env (e.g. DISCORD_GUILD_ID=) fall back to defaults.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_ignore_empty=True)

    database_url: str = "postgresql+asyncpg://threadlight:threadlight@localhost:5433/threadlight"

    discord_bot_token: str = ""
    discord_guild_id: int | None = None
    anthropic_api_key: str = ""
    voyage_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
