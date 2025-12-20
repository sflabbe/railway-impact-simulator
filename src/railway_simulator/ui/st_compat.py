"""Streamlit compatibility helpers.

This project historically used `width="stretch"` for several widgets/charts.
Streamlit does not accept `width=` for buttons, download buttons or charts.
Newer Streamlit versions support `use_container_width=True` for some widgets.

These helpers:
- map `width="stretch"` -> `use_container_width=True` when possible
- gracefully fallback when a widget does not support `use_container_width`
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def _coerce_container_width(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert legacy `width` kwarg to Streamlit's `use_container_width`.

    We mutate a shallow copy to avoid side-effects on caller dicts.
    """
    out = dict(kwargs)
    width = out.pop("width", None)
    if isinstance(width, str) and width.lower() in {"stretch", "container", "wide", "full"}:
        # Many Streamlit elements accept this, but not all.
        out.setdefault("use_container_width", True)
    return out


def safe_call(widget_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a Streamlit widget with best-effort compatibility."""
    kw = _coerce_container_width(kwargs)
    try:
        return widget_fn(*args, **kw)
    except TypeError:
        # Older Streamlit or widget that doesn't accept use_container_width
        kw.pop("use_container_width", None)
        return widget_fn(*args, **kw)


def safe_plotly_chart(st_mod: Any, fig: Any, **kwargs: Any) -> Any:
    return safe_call(st_mod.plotly_chart, fig, **kwargs)


def safe_button(st_mod: Any, label: str, **kwargs: Any) -> bool:
    return bool(safe_call(st_mod.button, label, **kwargs))


def safe_download_button(st_mod: Any, label: str, data: Any, file_name: str, mime: str, **kwargs: Any) -> Any:
    return safe_call(st_mod.download_button, label, data, file_name, mime, **kwargs)
