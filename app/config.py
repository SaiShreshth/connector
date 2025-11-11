from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # PostgreSQL
    pg_dsn: str | None = Field(default=None, alias="PG_DSN")
    pg_host: str | None = Field(default=None, alias="PG_HOST")
    pg_port: int | None = Field(default=None, alias="PG_PORT")
    pg_db: str | None = Field(default=None, alias="PG_DB")
    pg_user: str | None = Field(default=None, alias="PG_USER")
    pg_password: str | None = Field(default=None, alias="PG_PASSWORD")
    pg_sslmode: str | None = Field(default=None, alias="PG_SSLMODE")

    # Shopify
    shopify_shop_domain: str | None = Field(default=None, alias="SHOPIFY_SHOP_DOMAIN")
    shopify_access_token: str | None = Field(default=None, alias="SHOPIFY_ACCESS_TOKEN")
    shopify_storefront_access_token: str | None = Field(default=None, alias="SHOPIFY_STOREFRONT_ACCESS_TOKEN")
    shopify_api_version: str = Field(default="2023-10", alias="SHOPIFY_API_VERSION")
    shopify_timeout_seconds: float = Field(default=15.0, alias="SHOPIFY_TIMEOUT_SECONDS")
    shopify_log_response_bodies: bool = Field(default=True, alias="SHOPIFY_LOG_RESPONSE_BODIES")

    class Config:
        env_file = ".env"
        extra = "ignore"


def build_pg_dsn(settings: Settings) -> str | None:
    if settings.pg_dsn:
        return settings.pg_dsn
    if not all([settings.pg_host, settings.pg_port, settings.pg_db, settings.pg_user, settings.pg_password]):
        return None
    sslmode = settings.pg_sslmode or "prefer"
    return (
        f"postgresql://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}?sslmode={sslmode}"
    )


