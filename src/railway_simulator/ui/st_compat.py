"""Streamlit compatibility helpers.

Streamlit's width API has changed across versions:

* Older versions did **not** accept `width=` for many widgets and preferred
  `use_container_width=True`.
* Newer versions prefer `width="stretch"|"content"` and deprecate
  `use_container_width`.

These helpers accept calls using either style and try the best option for the
installed Streamlit version:

1) Try calling with `width=` if provided (or derived from `use_container_width`).
2) If that fails, map `width="stretch"` -> `use_container_width=True`.
3) If that still fails, call without width-related kwargs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


_WIDTH_STRETCH = {"stretch", "container", "wide", "full"}
_WIDTH_CONTENT = {"content", "auto"}


def _use_container_width_to_width(value: Any) -> str | None:
    if isinstance(value, bool):
        return "stretch" if value else "content"
    return None


def _coerce_to_width(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """If caller used `use_container_width`, map it to the new `width=` API."""
    out = dict(kwargs)
    if "width" not in out and "use_container_width" in out:
        w = _use_container_width_to_width(out.get("use_container_width"))
        if w is not None:
            out.pop("use_container_width", None)
            out["width"] = w
    return out


def _coerce_to_use_container_width(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """If caller used `width=`, map it to the legacy `use_container_width` API."""
    out = dict(kwargs)
    width = out.pop("width", None)
    if isinstance(width, str):
        w = width.lower()
        if w in _WIDTH_STRETCH:
            out.setdefault("use_container_width", True)
        elif w in _WIDTH_CONTENT:
            out.setdefault("use_container_width", False)
    return out


def safe_call(widget_fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a Streamlit widget with best-effort compatibility."""
    # Prefer new Streamlit API (width=) when possible to avoid deprecation noise.
    kw = _coerce_to_width(kwargs)
    try:
        return widget_fn(*args, **kw)
    except TypeError:
        # Older Streamlit / widget that doesn't accept width
        kw_legacy = _coerce_to_use_container_width(kwargs)
        try:
            return widget_fn(*args, **kw_legacy)
        except TypeError:
            # Last resort: drop width-related kwargs entirely.
            kw_last = dict(kwargs)
            kw_last.pop("width", None)
            kw_last.pop("use_container_width", None)
            return widget_fn(*args, **kw_last)


def safe_plotly_chart(st_mod: Any, fig: Any, **kwargs: Any) -> Any:
    return safe_call(st_mod.plotly_chart, fig, **kwargs)


def safe_button(st_mod: Any, label: str, **kwargs: Any) -> bool:
    return bool(safe_call(st_mod.button, label, **kwargs))


def safe_download_button(st_mod: Any, label: str, data: Any, file_name: str, mime: str, **kwargs: Any) -> Any:
    return safe_call(st_mod.download_button, label, data, file_name, mime, **kwargs)
