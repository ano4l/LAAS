from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _csv_env(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_env: str
    app_version: str
    cors_origins: list[str]
    storage_root: Path
    upload_dir: Path
    output_dir: Path
    database_path: Path
    max_upload_mb: int
    football_analytics_engine: str
    openai_api_key: str | None

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    storage_root = Path(os.getenv("STORAGE_ROOT", ROOT / "data" / "storage")).resolve()
    upload_dir = Path(os.getenv("UPLOAD_DIR", storage_root / "uploads")).resolve()
    output_dir = Path(os.getenv("OUTPUT_DIR", storage_root / "outputs")).resolve()
    database_path = Path(os.getenv("DATABASE_PATH", storage_root / "football_analytics.sqlite3")).resolve()

    for path in (storage_root, upload_dir, output_dir, database_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_env=os.getenv("APP_ENV", "development"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        cors_origins=_csv_env("CORS_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"),
        storage_root=storage_root,
        upload_dir=upload_dir,
        output_dir=output_dir,
        database_path=database_path,
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "4096")),
        football_analytics_engine="yolo_bytetrack",
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
    )


settings = load_settings()
