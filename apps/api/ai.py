from __future__ import annotations

from apps.api.config import settings


def provider_status() -> dict:
    return {
        "configured": bool(settings.openai_api_key),
        "provider": "openai" if settings.openai_api_key else None,
        "features": {
            "report_enrichment": bool(settings.openai_api_key),
            "coach_chat": bool(settings.openai_api_key),
            "scouting_summary": bool(settings.openai_api_key),
        },
        "message": "AI key configured" if settings.openai_api_key else "Set OPENAI_API_KEY to enable generated report enrichment.",
    }


def enrich_report(report: dict) -> dict:
    """Stable no-op boundary until an AI provider key is supplied."""
    status = provider_status()
    return {
        "provider": status,
        "report": report,
        "generated": False,
    }
