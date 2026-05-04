"""Streamlit wrapper for the TypeScript trade planner component."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import os

import streamlit.components.v1 as components


_COMPONENT_NAME = "trade_planner_component"
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"
_DEV_SERVER_URL = os.environ.get("TRADE_PLANNER_COMPONENT_URL")

if _DEV_SERVER_URL:
    _COMPONENT_FUNC = components.declare_component(
        _COMPONENT_NAME,
        url=_DEV_SERVER_URL,
    )
else:
    _COMPONENT_FUNC = components.declare_component(
        _COMPONENT_NAME,
        path=str(_FRONTEND_DIR),
    )


def render_trade_planner_component(
    *,
    props: Mapping[str, Any],
    key: str,
) -> dict[str, Any] | None:
    """Render the trade planner component and return the latest UI event payload."""
    return _COMPONENT_FUNC(
        key=key,
        default=None,
        **dict(props),
    )


def trade_planner_component_available() -> bool:
    """Return whether the checked-in component build is available locally."""
    return _DEV_SERVER_URL is not None or _FRONTEND_DIR.exists()

