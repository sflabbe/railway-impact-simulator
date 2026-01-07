"""Terminal live monitor.

Curses based full screen monitor inspired by htop.
Designed to be light, dependency free, and usable over SSH including mobile clients.

The engine can call a progress callback that pushes dict snapshots.
The CLI consumes those snapshots and renders a full screen monitor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Any, Optional, Iterable, Tuple, List

import math
import os
import queue
import time

try:
    import curses  # type: ignore
except Exception:  # pragma: no cover
    curses = None  # type: ignore


_SPARK = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = " .:-=+*#%@"

_BRAILLE_BASE = 0x2800
_BRAILLE_BITS = (
    (0x01, 0x08),
    (0x02, 0x10),
    (0x04, 0x20),
    (0x40, 0x80),
)


def _set_braille(mask_grid: List[List[int]], x_sub: int, y_sub: int) -> None:
    cell_x = x_sub // 2
    cell_y = y_sub // 4
    if cell_y < 0 or cell_y >= len(mask_grid):
        return
    row = mask_grid[cell_y]
    if cell_x < 0 or cell_x >= len(row):
        return
    dx = x_sub % 2
    dy = y_sub % 4
    row[cell_x] |= _BRAILLE_BITS[dy][dx]


def _draw_line_braille(mask_grid: List[List[int]], x0: int, y0: int, x1: int, y1: int) -> None:
    """Bresenham line in sub pixel coordinates."""
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x = x0
    y = y0
    while True:
        _set_braille(mask_grid, x, y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _render_braille_row(mask_row: List[int]) -> str:
    out = []
    for m in mask_row:
        if m:
            out.append(chr(_BRAILLE_BASE + m))
        else:
            out.append(" ")
    return "".join(out)


def _supports_unicode() -> bool:
    try:
        if curses is None:
            return False
        return bool(curses.has_unicode_support())
    except Exception:
        return False


def _fmt_num(x: float) -> str:
    """Compact numeric formatting for axes and headers."""
    x = float(x)
    ax = abs(x)
    if ax < 1e-12:
        return "0"
    if ax >= 1e6:
        return f"{x:.3e}"
    if ax >= 1e3:
        return f"{x:.4g}"
    if ax >= 1.0:
        return f"{x:.4g}"
    if ax >= 1e-3:
        return f"{x:.4g}"
    return f"{x:.3e}"


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    q = float(q)
    if q <= 0.0:
        return float(sorted_vals[0])
    if q >= 1.0:
        return float(sorted_vals[-1])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if hi <= lo:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _robust_minmax(vals: List[float], q_lo: float = 0.01, q_hi: float = 0.99) -> Tuple[float, float]:
    if not vals:
        return 0.0, 1.0
    if len(vals) < 6:
        return float(min(vals)), float(max(vals))
    sv = sorted(float(v) for v in vals)
    lo = _quantile(sv, q_lo)
    hi = _quantile(sv, q_hi)
    if abs(hi - lo) < 1e-30:
        return float(min(vals)), float(max(vals))
    return float(lo), float(hi)


def _downsample_max(x: List[float], y: List[float], width: int) -> Tuple[List[float], List[float]]:
    if width <= 1 or len(x) <= width:
        return x, y
    x0 = x[0]
    x1 = x[-1]
    if x1 <= x0:
        return x[:width], y[:width]
    out_x: List[float] = []
    out_y: List[float] = []
    n = len(x)
    for j in range(width):
        xa = x0 + (x1 - x0) * j / width
        xb = x0 + (x1 - x0) * (j + 1) / width
        best = None
        best_x = None
        for i in range(n):
            if x[i] < xa:
                continue
            if x[i] >= xb:
                break
            if best is None or y[i] > best:
                best = y[i]
                best_x = x[i]
        if best is None:
            k = min(n - 1, int(j * n / width))
            best = y[k]
            best_x = x[k]
        out_x.append(float(best_x))
        out_y.append(float(best))
    return out_x, out_y


def _downsample_last(x: List[float], y: List[float], width: int) -> Tuple[List[float], List[float]]:
    if width <= 1:
        if not x:
            return [], []
        return [x[-1]], [y[-1]]
    n = len(x)
    if n <= width:
        return x[:], y[:]
    xa = float(x[0])
    xb = float(x[-1])
    if xb <= xa:
        step = max(1, int(n / width))
        xs = [x[i] for i in range(0, n, step)]
        ys = [y[i] for i in range(0, n, step)]
        return xs[:width], ys[:width]
    out_x: List[float] = []
    out_y: List[float] = []
    for j in range(width):
        a = xa + (xb - xa) * (j / width)
        b = xa + (xb - xa) * ((j + 1) / width)
        last_y = None
        last_x = None
        for i in range(n):
            if x[i] < a:
                continue
            if x[i] >= b:
                break
            last_y = y[i]
            last_x = x[i]
        if last_y is None:
            k = min(n - 1, int((j + 1) * n / width))
            last_y = y[k]
            last_x = x[k]
        out_x.append(float(last_x))
        out_y.append(float(last_y))
    return out_x, out_y


def _safe_addstr(stdscr, y: int, x: int, s: str, attr: int = 0) -> None:
    """Add string to curses window without raising on small terminals."""
    try:
        h, w = stdscr.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return
        if x < 0:
            s = str(s)[-x:]
            x = 0
        else:
            s = str(s)
        if not s:
            return
        s = s[: max(0, w - x - 1)]
        if not s:
            return
        try:
            stdscr.addstr(y, x, s, attr)
        except Exception:
            stdscr.addstr(y, x, s)
    except Exception:
        return


def _sparkline(values, width: int, ascii_only: bool) -> str:
    if width <= 0:
        return ""
    if values is None or len(values) == 0:
        return " " * width

    v = list(values)
    if len(v) == 1:
        v = v * 2

    vmin = min(v)
    vmax = max(v)
    if abs(vmax - vmin) < 1e-30:
        vmax = vmin + 1.0

    chars = _SPARK_ASCII if ascii_only else _SPARK
    nlev = len(chars) - 1

    out = []
    n = len(v)
    for j in range(width):
        i0 = int(j * n / width)
        i1 = int((j + 1) * n / width)
        if i1 <= i0:
            i1 = min(i0 + 1, n)
        seg = v[i0:i1]
        y = max(seg)
        r = (y - vmin) / (vmax - vmin)
        k = int(round(r * nlev))
        k = max(0, min(nlev, k))
        out.append(chars[k])
    return "".join(out)


def _is_trivial(buf: Deque[float], eps: float = 1e-18) -> bool:
    for v in buf:
        if abs(float(v)) > eps:
            return False
    return True


@dataclass
class MonitorConfig:
    buffer_len: int = 600
    refresh_s: float = 0.05
    ascii_only: bool = False
    hold_on_done: bool = True


class HtopLikeMonitor:
    def __init__(
        self,
        updates: "queue.Queue[Dict[str, Any]]",
        done_event,
        title: str = "Railway impact simulator",
        cfg: Optional[MonitorConfig] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.updates = updates
        self.done_event = done_event
        self.title = title
        self.cfg = cfg or MonitorConfig()

        self.t0_wall = time.perf_counter()
        self.paused = False
        self.detached = False

        self.view_idx = 0
        self.last: Dict[str, Any] = {}

        self.window_modes_ms: List[Optional[float]] = [None, 200.0, 500.0, 1000.0, 2000.0]
        self.window_idx = 0

        self.show_contact_markers = True
        self.contact_edges_ms: Deque[float] = deque(maxlen=64)
        self._last_contact_active: Optional[bool] = None

        self.energy_normalize = True

        self.toast_msg = ""
        self.toast_until = 0.0

        env_out = os.environ.get("RIS_OUTPUT_DIR") or os.environ.get("RIS_MONITOR_OUTDIR")
        self.output_dir = output_dir or (Path(env_out) if env_out else None)

        self.buf_t_ms: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.buf_force: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_pen: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_ratio: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_ratio_ppm: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.buf_ekin: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_ediss: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_enum: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.buf_emech: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_epot: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_wext: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.peak_ekin = 0.0
        self.peak_ediss = 0.0
        self.peak_enum = 0.0

        self.peak_force = 0.0
        self.peak_pen = 0.0
        self.peak_ratio = 0.0
        self.peak_ratio_ppm = 0.0

    def _consume_updates(self) -> None:
        latest = None
        while True:
            try:
                latest = self.updates.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        self.last = latest
        if self.paused:
            return

        f = float(latest.get("impact_force_MN", 0.0) or 0.0)
        p = float(latest.get("penetration_mm", 0.0) or 0.0)
        r = float(latest.get("E_num_ratio", 0.0) or 0.0)

        ek = float(latest.get("E_kin_J", 0.0) or 0.0)
        ed = float(latest.get("E_diss_total_J", 0.0) or 0.0)
        en = float(latest.get("E_num_J", 0.0) or 0.0)

        em = float(latest.get("E_mech_J", 0.0) or 0.0)
        ep = float(latest.get("E_pot_J", 0.0) or 0.0)
        wx = float(latest.get("W_ext_J", 0.0) or 0.0)

        t_ms = float(latest.get("t", 0.0) or 0.0) * 1000.0

        self.buf_t_ms.append(t_ms)

        self.buf_force.append(f)
        self.buf_pen.append(p)
        self.buf_ratio.append(r)
        self.buf_ratio_ppm.append(r * 1e6)

        self.buf_ekin.append(ek)
        self.buf_ediss.append(ed)
        self.buf_enum.append(en)

        self.buf_emech.append(em)
        self.buf_epot.append(ep)
        self.buf_wext.append(wx)

        self.peak_force = max(self.peak_force, f)
        self.peak_pen = max(self.peak_pen, p)
        self.peak_ratio = max(self.peak_ratio, r)
        self.peak_ratio_ppm = max(self.peak_ratio_ppm, r * 1e6)

        self.peak_ekin = max(self.peak_ekin, ek)
        self.peak_ediss = max(self.peak_ediss, ed)
        self.peak_enum = max(self.peak_enum, abs(en))

        contact_active = bool(latest.get("contact_active", latest.get("contact_active", False)))
        if self._last_contact_active is None:
            self._last_contact_active = contact_active
        else:
            if contact_active != self._last_contact_active:
                self.contact_edges_ms.append(t_ms)
            self._last_contact_active = contact_active

    def _window_label(self) -> str:
        win_ms = self.window_modes_ms[self.window_idx]
        if win_ms is None:
            return "full"
        if win_ms >= 1000.0:
            return f"{int(round(win_ms / 1000.0))}s"
        return f"{int(round(win_ms))}ms"

    def _window_start_idx(self, xs_ms: List[float]) -> int:
        if not xs_ms:
            return 0
        win_ms = self.window_modes_ms[self.window_idx]
        if win_ms is None:
            return 0
        t_end = float(xs_ms[-1])
        t_start = t_end - float(win_ms)
        i0 = 0
        n = len(xs_ms)
        while i0 < n and float(xs_ms[i0]) < t_start:
            i0 += 1
        return i0

    def _aligned_window(
        self,
        xs_ms: List[float],
        ys: List[float],
        i0: int,
    ) -> Tuple[List[float], List[float]]:
        n = min(len(xs_ms), len(ys))
        if n <= 0:
            return [], []
        xs_ms = xs_ms[-n:]
        ys = ys[-n:]
        i0 = max(0, min(i0, n - 1))
        return xs_ms[i0:], ys[i0:]

    def _marker_cols(
        self,
        markers_ms: Optional[Iterable[float]],
        x_base_ms: float,
        x_end_ms: float,
        plot_w: int,
    ) -> List[int]:
        if not markers_ms:
            return []
        if plot_w <= 1:
            return []
        span = float(x_end_ms - x_base_ms)
        if span <= 1e-12:
            return []
        cols: List[int] = []
        for t_abs in markers_ms:
            x_rel = float(t_abs) - float(x_base_ms)
            if x_rel < 0.0 or x_rel > span:
                continue
            col = int(round(x_rel / span * (plot_w - 1)))
            col = max(0, min(plot_w - 1, col))
            cols.append(col)
        cols = sorted(set(cols))
        return cols

    def _draw_contact_markers(
        self,
        win,
        top: int,
        left: int,
        plot_h: int,
        plot_w: int,
        cols: List[int],
    ) -> None:
        if curses is None:
            return
        if not cols or plot_h <= 0 or plot_w <= 0:
            return
        try:
            attr = curses.A_DIM
        except Exception:
            attr = 0
        ch = "│" if not self.cfg.ascii_only else "|"
        for c in cols:
            x = left + c
            for r in range(plot_h):
                _safe_addstr(win, top + r, x, ch, attr)

    def _draw_plot_core(
        self,
        win,
        top: int,
        left: int,
        height: int,
        width: int,
        x_ms: List[float],
        y_vals: List[float],
        y_unit: str,
        color_pair: int,
        downsample_kind: str = "max",
        show_y_labels: bool = True,
        markers_ms: Optional[Iterable[float]] = None,
        robust_scale: bool = True,
    ) -> None:
        if curses is None:
            return
        if height < 3 or width < 10:
            return

        if not x_ms or not y_vals:
            return

        n = min(len(x_ms), len(y_vals))
        if n < 2:
            return
        x_ms = x_ms[-n:]
        y_vals = y_vals[-n:]

        x_base = float(x_ms[0])
        xs = [float(v) - x_base for v in x_ms]
        ys = [float(v) for v in y_vals]

        ylab_w = 9 if show_y_labels else 0
        plot_h = max(1, height)
        plot_w = max(1, width - ylab_w)

        if robust_scale:
            y_min, y_max = _robust_minmax(ys)
        else:
            y_min = float(min(ys))
            y_max = float(max(ys))

        if y_min >= 0.0:
            y_min = 0.0

        if abs(y_max - y_min) < 1e-30:
            y_max = y_min + 1.0
        else:
            span = y_max - y_min
            y_min = y_min - 0.03 * span
            y_max = y_max + 0.03 * span
            if y_min >= 0.0:
                y_min = 0.0

        if ylab_w > 0:
            for frac in (1.0, 0.5, 0.0):
                yy = y_min + frac * (y_max - y_min)
                row = int(round((1.0 - frac) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                lab = _fmt_num(yy).rjust(ylab_w - 1)
                _safe_addstr(win, top + row, left + 0, lab)

        attr = 0
        try:
            if curses.has_colors() and color_pair > 0:
                attr = curses.color_pair(color_pair)
        except Exception:
            attr = 0

        cols = []
        if self.show_contact_markers and markers_ms is not None:
            cols = self._marker_cols(markers_ms, x_base, float(x_ms[-1]), plot_w)
            self._draw_contact_markers(win, top, left + ylab_w, plot_h, plot_w, cols)

        ascii_only = bool(self.cfg.ascii_only)
        if ascii_only:
            plot_char = "#"
            line_char = "|"
            if downsample_kind == "last":
                _, ys_ds = _downsample_last(xs, ys, plot_w)
            else:
                _, ys_ds = _downsample_max(xs, ys, plot_w)

            prev_row = None
            prev_col = None
            for j in range(len(ys_ds)):
                col = j
                yv = float(ys_ds[j])
                rr = (yv - y_min) / (y_max - y_min)
                rr = max(0.0, min(1.0, rr))
                row = int(round((1.0 - rr) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))

                wy = top + row
                wx = left + ylab_w + col
                _safe_addstr(win, wy, wx, plot_char, attr)

                if prev_row is not None and prev_col is not None and col != prev_col:
                    y0 = min(prev_row, row)
                    y1 = max(prev_row, row)
                    if y1 - y0 >= 2:
                        for rrow in range(y0 + 1, y1):
                            _safe_addstr(win, top + rrow, left + ylab_w + col, line_char, attr)
                prev_row = row
                prev_col = col
            return

        w_sub = max(2, plot_w * 2)
        h_sub = max(4, plot_h * 4)

        if downsample_kind == "last":
            _, ys_ds = _downsample_last(xs, ys, w_sub)
        else:
            _, ys_ds = _downsample_max(xs, ys, w_sub)

        if not ys_ds:
            return

        mask_grid: List[List[int]] = [[0 for _ in range(plot_w)] for _ in range(plot_h)]

        npts = len(ys_ds)
        for k in range(npts):
            yv = float(ys_ds[k])
            rr = (yv - y_min) / (y_max - y_min)
            rr = max(0.0, min(1.0, rr))
            x_sub = 0 if npts <= 1 else int(round(k * (w_sub - 1) / (npts - 1)))
            y_sub = int(round((1.0 - rr) * (h_sub - 1)))
            if k == 0:
                x_prev = x_sub
                y_prev = y_sub
                _set_braille(mask_grid, x_sub, y_sub)
            else:
                _draw_line_braille(mask_grid, x_prev, y_prev, x_sub, y_sub)
                x_prev = x_sub
                y_prev = y_sub

        for r in range(plot_h):
            row_txt = _render_braille_row(mask_grid[r])
            _safe_addstr(win, top + r, left + ylab_w, row_txt, attr)

    def _draw_plot_multi_core(
        self,
        win,
        top: int,
        left: int,
        height: int,
        width: int,
        x_ms: List[float],
        series: List[Tuple[str, List[float], int]],
        base_unit: str = "J",
        normalize: bool = False,
        markers_ms: Optional[Iterable[float]] = None,
    ) -> None:
        if curses is None:
            return
        if height < 4 or width < 18:
            return
        if not x_ms:
            return

        n = len(x_ms)
        if n < 2:
            return

        ys_list: List[List[float]] = []
        for _label, ys, _cp in series:
            ys_list.append(list(ys))

        n = min([n] + [len(ys) for ys in ys_list]) if ys_list else n
        if n < 2:
            return

        x_ms = x_ms[-n:]
        ys_list = [ys[-n:] for ys in ys_list]

        x_base = float(x_ms[0])
        xs = [float(v) - x_base for v in x_ms]

        ylab_w = 9
        if width - ylab_w < 6:
            ylab_w = 0
        plot_w = max(1, width - ylab_w)

        legend_y = top
        plot_top = top + 1
        plot_h = height - 1
        if plot_h < 2:
            return

        ascii_only = bool(self.cfg.ascii_only)

        # Contact markers
        if self.show_contact_markers and markers_ms is not None:
            cols = self._marker_cols(markers_ms, x_base, float(x_ms[-1]), plot_w)
            self._draw_contact_markers(win, plot_top, left + ylab_w, plot_h, plot_w, cols)

        if normalize:
            ys_plot_list: List[List[float]] = []
            legend_texts: List[str] = []
            unit_txt = "norm"

            for (label, _ys, _cp), ys in zip(series, ys_list):
                absmax = 0.0
                if ys:
                    absmax = float(max(abs(float(v)) for v in ys))
                if absmax < 1e-30:
                    absmax = 1.0
                ys_plot_list.append([float(v) / absmax for v in ys])

                scale_i = 1.0
                unit_i = base_unit
                if absmax >= 1e6:
                    scale_i = 1e6
                    unit_i = "MJ"
                elif absmax >= 1e3:
                    scale_i = 1e3
                    unit_i = "kJ"
                last_i = float(ys[-1]) / scale_i if ys else 0.0
                legend_texts.append(f"{label} {_fmt_num(last_i)} {unit_i} ")

            y_min_s = -1.05
            y_max_s = 1.05

            if ylab_w > 0:
                for frac, yy in ((1.0, 1.0), (0.5, 0.0), (0.0, -1.0)):
                    row = int(round((1.0 - frac) * (plot_h - 1)))
                    row = max(0, min(plot_h - 1, row))
                    lab = _fmt_num(yy).rjust(ylab_w - 1)
                    _safe_addstr(win, plot_top + row, left + 0, lab)

            xoff = left + ylab_w + 1
            for seg, (_label, _ys, cp) in zip(legend_texts, series):
                attr = 0
                try:
                    if curses.has_colors() and cp > 0:
                        attr = curses.color_pair(cp)
                except Exception:
                    attr = 0
                _safe_addstr(win, legend_y, xoff, seg, attr)
                xoff += len(seg)
                if xoff >= left + width - 2:
                    break
            _safe_addstr(win, legend_y, left + width - len(unit_txt) - 1, unit_txt)

            # Zero baseline
            try:
                frac0 = (0.0 - y_min_s) / (y_max_s - y_min_s)
                frac0 = max(0.0, min(1.0, frac0))
                row0 = int(round((1.0 - frac0) * (plot_h - 1)))
                row0 = max(0, min(plot_h - 1, row0))
                win.hline(plot_top + row0, left + ylab_w, curses.ACS_HLINE, plot_w)
            except Exception:
                pass

            def _draw_series(ys_scaled: List[float], cp: int) -> None:
                attr = 0
                try:
                    if curses.has_colors() and cp > 0:
                        attr = curses.color_pair(cp)
                except Exception:
                    attr = 0

                if ascii_only:
                    _, ys_ds = _downsample_last(xs, ys_scaled, plot_w)
                    for j, yv in enumerate(ys_ds):
                        rr = (float(yv) - y_min_s) / (y_max_s - y_min_s)
                        rr = max(0.0, min(1.0, rr))
                        row = int(round((1.0 - rr) * (plot_h - 1)))
                        row = max(0, min(plot_h - 1, row))
                        _safe_addstr(win, plot_top + row, left + ylab_w + j, "#", attr)
                    return

                w_sub = max(2, plot_w * 2)
                h_sub = max(4, plot_h * 4)

                _, ys_ds = _downsample_last(xs, ys_scaled, w_sub)
                if not ys_ds:
                    return

                mask_grid: List[List[int]] = [[0 for _ in range(plot_w)] for _ in range(plot_h)]

                npts = len(ys_ds)
                for k in range(npts):
                    yv = float(ys_ds[k])
                    rr = (yv - y_min_s) / (y_max_s - y_min_s)
                    rr = max(0.0, min(1.0, rr))
                    x_sub = 0 if npts <= 1 else int(round(k * (w_sub - 1) / (npts - 1)))
                    y_sub = int(round((1.0 - rr) * (h_sub - 1)))
                    if k == 0:
                        x_prev = x_sub
                        y_prev = y_sub
                        _set_braille(mask_grid, x_sub, y_sub)
                    else:
                        _draw_line_braille(mask_grid, x_prev, y_prev, x_sub, y_sub)
                        x_prev = x_sub
                        y_prev = y_sub

                for r in range(plot_h):
                    row_txt = _render_braille_row(mask_grid[r])
                    c = 0
                    while c < plot_w:
                        if row_txt[c] == " ":
                            c += 1
                            continue
                        c0 = c
                        while c < plot_w and row_txt[c] != " ":
                            c += 1
                        seg = row_txt[c0:c]
                        _safe_addstr(win, plot_top + r, left + ylab_w + c0, seg, attr)

            for ys_scaled, (_label, _ys, cp) in zip(ys_plot_list, series):
                _draw_series(ys_scaled, cp)

            return

        # Absolute mode
        y_min = None
        y_max = None
        abs_max = 0.0
        for ys in ys_list:
            if not ys:
                continue
            abs_max = max(abs_max, float(max(abs(v) for v in ys)))
            mn0, mx0 = _robust_minmax([float(v) for v in ys])
            y_min = mn0 if y_min is None else min(y_min, mn0)
            y_max = mx0 if y_max is None else max(y_max, mx0)
        if y_min is None or y_max is None:
            return
        if abs(y_max - y_min) < 1e-30:
            y_max = y_min + 1.0

        scale = 1.0
        unit = base_unit
        if abs_max >= 1e6:
            scale = 1e6
            unit = "MJ"
        elif abs_max >= 1e3:
            scale = 1e3
            unit = "kJ"

        y_min_s = float(y_min) / scale
        y_max_s = float(y_max) / scale

        if y_min_s >= 0.0:
            y_min_s = 0.0

        span = y_max_s - y_min_s
        if abs(span) < 1e-30:
            y_max_s = y_min_s + 1.0
            span = y_max_s - y_min_s

        y_min_s = y_min_s - 0.03 * span
        y_max_s = y_max_s + 0.03 * span
        if y_min_s >= 0.0:
            y_min_s = 0.0
        if abs(y_max_s - y_min_s) < 1e-30:
            y_max_s = y_min_s + 1.0

        if ylab_w > 0:
            for frac in (1.0, 0.5, 0.0):
                yy = y_min_s + frac * (y_max_s - y_min_s)
                row = int(round((1.0 - frac) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                lab = _fmt_num(yy).rjust(ylab_w - 1)
                _safe_addstr(win, plot_top + row, left + 0, lab)

        xoff = left + ylab_w + 1
        for s_idx, (label, _ys, cp) in enumerate(series):
            try:
                last_v = float(ys_list[s_idx][-1]) / float(scale)
            except Exception:
                last_v = None
            seg = f"{label} " if last_v is None else f"{label} {_fmt_num(last_v)} "

            attr = 0
            try:
                if curses.has_colors() and cp > 0:
                    attr = curses.color_pair(cp)
            except Exception:
                attr = 0
            _safe_addstr(win, legend_y, xoff, seg, attr)
            xoff += len(seg)
            if xoff >= left + width - 2:
                break

        unit_txt = f"{unit}"
        _safe_addstr(win, legend_y, left + width - len(unit_txt) - 1, unit_txt)

        def _draw_braille_series(ys_scaled: List[float], cp: int) -> None:
            attr = 0
            try:
                if curses.has_colors() and cp > 0:
                    attr = curses.color_pair(cp)
            except Exception:
                attr = 0

            if ascii_only:
                _, ys_ds = _downsample_last(xs, ys_scaled, plot_w)
                for j, yv in enumerate(ys_ds):
                    rr = (float(yv) - y_min_s) / (y_max_s - y_min_s)
                    rr = max(0.0, min(1.0, rr))
                    row = int(round((1.0 - rr) * (plot_h - 1)))
                    row = max(0, min(plot_h - 1, row))
                    _safe_addstr(win, plot_top + row, left + ylab_w + j, "*", attr)
                return

            w_sub = max(2, plot_w * 2)
            h_sub = max(4, plot_h * 4)

            _, ys_ds = _downsample_last(xs, ys_scaled, w_sub)
            if not ys_ds:
                return

            mask_grid: List[List[int]] = [[0 for _ in range(plot_w)] for _ in range(plot_h)]

            npts = len(ys_ds)
            for k in range(npts):
                yv = float(ys_ds[k])
                rr = (yv - y_min_s) / (y_max_s - y_min_s)
                rr = max(0.0, min(1.0, rr))
                x_sub = 0 if npts <= 1 else int(round(k * (w_sub - 1) / (npts - 1)))
                y_sub = int(round((1.0 - rr) * (h_sub - 1)))
                if k == 0:
                    x_prev = x_sub
                    y_prev = y_sub
                    _set_braille(mask_grid, x_sub, y_sub)
                else:
                    _draw_line_braille(mask_grid, x_prev, y_prev, x_sub, y_sub)
                    x_prev = x_sub
                    y_prev = y_sub

            for r in range(plot_h):
                row_txt = _render_braille_row(mask_grid[r])
                c = 0
                while c < plot_w:
                    if row_txt[c] == " ":
                        c += 1
                        continue
                    c0 = c
                    while c < plot_w and row_txt[c] != " ":
                        c += 1
                    seg = row_txt[c0:c]
                    _safe_addstr(win, plot_top + r, left + ylab_w + c0, seg, attr)

        for s_idx, (_label, _ys, cp) in enumerate(series):
            ys = ys_list[s_idx]
            ys_scaled = [float(v) / scale for v in ys]
            _draw_braille_series(ys_scaled, cp)

    def _draw_box_plot(
        self,
        stdscr,
        top: int,
        left: int,
        height: int,
        width: int,
        title: str,
        x_ms: List[float],
        y_vals: List[float],
        last_val: float,
        peak_val: float,
        y_unit: str,
        color_pair: int = 0,
        markers_ms: Optional[Iterable[float]] = None,
    ) -> None:
        if curses is None:
            return
        if height < 5 or width < 20:
            return

        try:
            win = stdscr.derwin(height, width, top, left)
        except Exception:
            return

        try:
            win.erase()
            win.box()
        except Exception:
            return

        attr_plot = 0
        try:
            if curses is not None and color_pair and curses.has_colors():
                attr_plot = curses.color_pair(int(color_pair))
        except Exception:
            attr_plot = 0

        inner_h = height - 2
        inner_w = width - 2
        if inner_h < 3 or inner_w < 10:
            return

        ylab_w = 0
        if inner_w >= 52:
            ylab_w = 8

        plot_w = max(1, inner_w - ylab_w)
        plot_h = max(1, inner_h)

        header = f" {title}  last { _fmt_num(last_val) } {y_unit}  peak { _fmt_num(peak_val) } {y_unit} "
        header = header[: max(0, width - 2)]
        _safe_addstr(win, 0, 1, header)

        n = min(len(x_ms), len(y_vals))
        if n < 2:
            return
        x_ms = x_ms[-n:]
        y_vals = y_vals[-n:]

        x_base = float(x_ms[0])
        xs = [float(v) - x_base for v in x_ms]
        ys = [float(v) for v in y_vals]

        x0 = 0.0
        x1 = float(xs[-1])
        if x1 <= x0:
            x1 = x0 + 1.0

        y_min, y_max = _robust_minmax(ys)
        if y_min >= 0.0:
            y_min = 0.0
        if abs(y_max - y_min) < 1e-30:
            y_max = y_min + 1.0
        else:
            span = y_max - y_min
            y_min = y_min - 0.03 * span
            y_max = y_max + 0.03 * span
            if y_min >= 0.0:
                y_min = 0.0

        if ylab_w > 0:
            for frac in (1.0, 0.5, 0.0):
                yy = y_min + frac * (y_max - y_min)
                row = int(round((1.0 - frac) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                lab = _fmt_num(yy).rjust(ylab_w - 1)
                _safe_addstr(win, 1 + row, 1, lab)

        if self.show_contact_markers and markers_ms is not None:
            cols = self._marker_cols(markers_ms, x_base, float(x_ms[-1]), plot_w)
            self._draw_contact_markers(win, 1, 1 + ylab_w, plot_h, plot_w, cols)

        ascii_only = bool(self.cfg.ascii_only)

        if ascii_only:
            _, ys_ds = _downsample_max(xs, ys, plot_w)
            prev_row = None
            prev_col = None
            for j in range(len(ys_ds)):
                col = j
                yv = float(ys_ds[j])
                r = (yv - y_min) / (y_max - y_min)
                r = max(0.0, min(1.0, r))
                row = int(round((1.0 - r) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                _safe_addstr(win, 1 + row, 1 + ylab_w + col, "#", attr_plot)
                if prev_row is not None and prev_col is not None and col == prev_col + 1 and row != prev_row:
                    a = min(row, prev_row)
                    b = max(row, prev_row)
                    for rr in range(a, b + 1):
                        _safe_addstr(win, 1 + rr, 1 + ylab_w + col, "|", attr_plot)
                prev_row = row
                prev_col = col
        else:
            w_sub = max(2, plot_w * 2)
            h_sub = max(4, plot_h * 4)
            _, ys_ds = _downsample_max(xs, ys, w_sub)
            if ys_ds:
                mask_grid: List[List[int]] = [[0 for _ in range(plot_w)] for _ in range(plot_h)]
                npts = len(ys_ds)
                for k in range(npts):
                    yv = float(ys_ds[k])
                    rr = (yv - y_min) / (y_max - y_min)
                    rr = max(0.0, min(1.0, rr))
                    x_sub = 0 if npts <= 1 else int(round(k * (w_sub - 1) / (npts - 1)))
                    y_sub = int(round((1.0 - rr) * (h_sub - 1)))
                    if k == 0:
                        x_prev = x_sub
                        y_prev = y_sub
                        _set_braille(mask_grid, x_sub, y_sub)
                    else:
                        _draw_line_braille(mask_grid, x_prev, y_prev, x_sub, y_sub)
                        x_prev = x_sub
                        y_prev = y_sub
                for r in range(plot_h):
                    row_txt = _render_braille_row(mask_grid[r])
                    _safe_addstr(win, 1 + r, 1 + ylab_w, row_txt, attr_plot)

        if width >= 36:
            xm = 0.5 * (x0 + x1)
            xlab = f" {x0:.0f} ms   {xm:.0f} ms   {x1:.0f} ms "
        else:
            xlab = f" {x1:.0f} ms "
        xlab = xlab[: max(0, width - 2)]
        _safe_addstr(win, height - 1, 1, xlab)

    def _draw_energy_panel(
        self,
        stdscr,
        top: int,
        left: int,
        height: int,
        width: int,
        x_ms: List[float],
        ratio_vals_ppm: List[float],
        last_ratio_ppm: float,
        peak_ratio_ppm: float,
        series: List[Tuple[str, List[float], int]],
        markers_ms: Optional[Iterable[float]] = None,
    ) -> None:
        if curses is None:
            return
        if height < 7 or width < 40:
            return
        try:
            win = stdscr.derwin(height, width, top, left)
        except Exception:
            return
        try:
            win.erase()
            win.box()
        except Exception:
            return

        title = "energy balance"
        mode = "norm" if self.energy_normalize else "abs"
        stats = f"Eb ppm  last { _fmt_num(last_ratio_ppm) }  peak { _fmt_num(peak_ratio_ppm) }  {mode}"
        header = f" {title}  {stats} "
        header = header[: max(0, width - 2)]
        _safe_addstr(win, 0, 2, header)

        inner_h = height - 2
        inner_w = width - 2
        if inner_h < 5 or inner_w < 10:
            return

        ratio_h = max(3, int(round(inner_h * 0.35)))
        ratio_h = min(ratio_h, inner_h - 2)
        ener_h = inner_h - ratio_h - 1

        sep_y = 1 + ratio_h
        if sep_y < height - 1:
            try:
                win.hline(sep_y, 1, curses.ACS_HLINE, width - 2)
            except Exception:
                pass

        self._draw_plot_core(
            win,
            top=1,
            left=1,
            height=ratio_h,
            width=width - 2,
            x_ms=x_ms,
            y_vals=ratio_vals_ppm,
            y_unit="ppm",
            color_pair=3,
            downsample_kind="max",
            show_y_labels=True,
            markers_ms=markers_ms,
            robust_scale=True,
        )

        if ener_h < 3:
            return

        self._draw_plot_multi_core(
            win,
            top=sep_y + 1,
            left=1,
            height=ener_h,
            width=width - 2,
            x_ms=x_ms,
            series=series,
            base_unit="J",
            normalize=bool(self.energy_normalize),
            markers_ms=markers_ms,
        )

    def _maybe_toast(self, now: float) -> str:
        if self.toast_msg and now < self.toast_until:
            return self.toast_msg
        return ""

    def _save_screenshot(self, stdscr) -> Optional[Path]:
        if curses is None:
            return None
        h, w = stdscr.getmaxyx()

        out_dir = self.output_dir
        if out_dir is None:
            out_dir = Path(os.getcwd())

        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            out_dir = Path(os.getcwd())

        ts = time.strftime("%Y%m%d_%H%M%S")
        view_name = "impact" if int(getattr(self, "view_idx", 0) or 0) == 0 else "energy"
        fname = f"monitor_{view_name}_{ts}.txt"
        path = out_dir / fname

        lines: List[str] = []
        for y in range(h):
            try:
                raw = stdscr.instr(y, 0, max(0, w - 1))
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="ignore")
                else:
                    line = str(raw)
            except Exception:
                line = ""
            lines.append(line.rstrip())

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            return None
        return path

    def _render(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        now = time.perf_counter()
        wall = now - self.t0_wall

        step = int(self.last.get("step", 0) or 0)
        n_steps = int(self.last.get("n_steps", 0) or 0)
        t = float(self.last.get("t", 0.0) or 0.0)
        dt = float(self.last.get("dt", 0.0) or 0.0)
        iters = int(self.last.get("iters", 0) or 0)
        solver = str(self.last.get("solver", ""))
        max_res = float(self.last.get("max_residual", 0.0) or 0.0)
        contact = bool(self.last.get("contact_active", False))

        f = float(self.last.get("impact_force_MN", 0.0) or 0.0)
        p = float(self.last.get("penetration_mm", 0.0) or 0.0)
        v = float(self.last.get("velocity_m_s", 0.0) or 0.0)
        a = float(self.last.get("acceleration_g", 0.0) or 0.0)
        ratio = float(self.last.get("E_num_ratio", 0.0) or 0.0)

        pct = 0.0
        if n_steps > 0:
            pct = 100.0 * float(step) / float(n_steps)
        if self.done_event.is_set():
            pct = 100.0

        status = "running"
        if self.done_event.is_set():
            status = "done"
        if self.paused:
            status = "paused"
        if self.detached:
            status = "detached"

        header = f"{self.title}  status {status}"
        _safe_addstr(stdscr, 0, 0, header)

        line1 = f"step {step}/{n_steps}  {pct:5.1f}%   t {t:9.4f} s   dt {dt:.2e}   wall {wall:7.1f} s"
        _safe_addstr(stdscr, 1, 0, line1)

        line2 = f"solver {solver}   iters {iters}   max resid {max_res:.2e}   contact {int(contact)}"
        _safe_addstr(stdscr, 2, 0, line2)

        line3 = f"F {f:9.3f} MN   pen {p:9.3f} mm   v {v:9.3f} m/s   a {a:9.3f} g   Eb {ratio:9.3e}"
        _safe_addstr(stdscr, 3, 0, line3)

        # Windowed buffers
        xs_full = list(self.buf_t_ms)
        if xs_full:
            i0 = self._window_start_idx(xs_full)
        else:
            i0 = 0

        xs_win, force_win = self._aligned_window(xs_full, list(self.buf_force), i0)
        _xs_win2, pen_win = self._aligned_window(xs_full, list(self.buf_pen), i0)
        _xs_win3, ratio_ppm_win = self._aligned_window(xs_full, list(self.buf_ratio_ppm), i0)

        _xs_win4, ek_win = self._aligned_window(xs_full, list(self.buf_ekin), i0)
        _xs_win5, ed_win = self._aligned_window(xs_full, list(self.buf_ediss), i0)
        _xs_win6, en_win = self._aligned_window(xs_full, list(self.buf_enum), i0)
        _xs_win7, em_win = self._aligned_window(xs_full, list(self.buf_emech), i0)
        _xs_win8, ep_win = self._aligned_window(xs_full, list(self.buf_epot), i0)
        _xs_win9, wx_win = self._aligned_window(xs_full, list(self.buf_wext), i0)

        markers = list(self.contact_edges_ms) if self.show_contact_markers else []

        # Plots area
        y_top = 4
        footer_h = 1
        avail_h = max(0, h - footer_h - y_top)
        gap = 1

        view_name = "impact" if int(getattr(self, "view_idx", 0) or 0) == 0 else "energy"

        # Tiny terminals fallback
        if avail_h < 12 or w < 60:
            y = y_top + 1
            plot_w = max(10, w - 28)

            def plot_row(label: str, buf, last_val: float, peak_val: float, y_row: int) -> None:
                spark = _sparkline(buf, plot_w, self.cfg.ascii_only)
                left_txt = f"{label:<10}"
                right_txt = f" last {last_val:9.3g}  peak {peak_val:9.3g}"
                _safe_addstr(stdscr, y_row, 0, left_txt)
                _safe_addstr(stdscr, y_row, 11, spark)
                _safe_addstr(stdscr, y_row, 11 + plot_w + 1, right_txt)

            if view_name == "impact":
                if y < h:
                    plot_row("force MN", force_win, f, self.peak_force, y)
                if y + 1 < h:
                    plot_row("pen mm", pen_win, p, self.peak_pen, y + 1)
                if y + 2 < h:
                    plot_row("Eb ppm", ratio_ppm_win, ratio * 1e6, self.peak_ratio_ppm, y + 2)
            else:
                if y < h:
                    plot_row("Eb ppm", ratio_ppm_win, ratio * 1e6, self.peak_ratio_ppm, y)
                if y + 1 < h:
                    plot_row("E kin J", ek_win, float(self.last.get("E_kin_J", 0.0) or 0.0), self.peak_ekin, y + 1)
                if y + 2 < h:
                    plot_row("E diss J", ed_win, float(self.last.get("E_diss_total_J", 0.0) or 0.0), self.peak_ediss, y + 2)
                if y + 3 < h:
                    plot_row("E num J", en_win, float(self.last.get("E_num_J", 0.0) or 0.0), self.peak_enum, y + 3)
        else:
            if view_name == "impact":
                n_pan = 2
                pan_h = int((avail_h - (n_pan - 1) * gap) / n_pan)
                if pan_h < 6:
                    pan_h = 6
                y0 = y_top
                h0 = pan_h
                h1 = max(6, avail_h - h0 - gap)

                self._draw_box_plot(
                    stdscr,
                    top=y0,
                    left=0,
                    height=h0,
                    width=w,
                    title="impact force",
                    x_ms=xs_win,
                    y_vals=force_win,
                    last_val=f,
                    peak_val=self.peak_force,
                    y_unit="MN",
                    color_pair=1,
                    markers_ms=markers,
                )
                y0 = y0 + h0 + gap
                self._draw_box_plot(
                    stdscr,
                    top=y0,
                    left=0,
                    height=h1,
                    width=w,
                    title="indentation",
                    x_ms=xs_win,
                    y_vals=pen_win,
                    last_val=p,
                    peak_val=self.peak_pen,
                    y_unit="mm",
                    color_pair=2,
                    markers_ms=markers,
                )
            else:
                series: List[Tuple[str, List[float], int]] = [
                    ("Ekin", ek_win, 1),
                    ("Ediss", ed_win, 2),
                    ("Enum", en_win, 4),
                ]
                if not _is_trivial(self.buf_emech):
                    series.append(("Emech", em_win, 6))
                if not _is_trivial(self.buf_epot):
                    series.append(("Epot", ep_win, 3))
                if not _is_trivial(self.buf_wext):
                    series.append(("Wext", wx_win, 5))

                self._draw_energy_panel(
                    stdscr,
                    top=y_top,
                    left=0,
                    height=avail_h,
                    width=w,
                    x_ms=xs_win,
                    ratio_vals_ppm=ratio_ppm_win,
                    last_ratio_ppm=ratio * 1e6,
                    peak_ratio_ppm=self.peak_ratio_ppm,
                    series=series,
                    markers_ms=markers,
                )

        # Footer
        win_txt = self._window_label()
        view_txt = f"view {view_name}"
        contact_txt = "contact on" if self.show_contact_markers else "contact off"
        energy_txt = "norm" if self.energy_normalize else "abs"
        toast = self._maybe_toast(now)

        if view_name == "energy":
            footer = f"keys q quit  v view  w win {win_txt}  c {contact_txt}  n {energy_txt}  s save  d detach  p pause"
        else:
            footer = f"keys q quit  v view  w win {win_txt}  c {contact_txt}  s save  d detach  p pause"

        if toast:
            footer = (footer + "  " + toast)

        if self.done_event.is_set() and self.cfg.hold_on_done:
            footer = "done  " + footer

        _safe_addstr(stdscr, h - 1, 0, footer)
        stdscr.refresh()

    def run(self) -> None:
        if curses is None:
            raise RuntimeError("curses is not available on this platform")
        curses.wrapper(self._loop)

    def _loop(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(max(1, self.cfg.refresh_s * 1000.0)))

        try:
            if curses.has_colors():
                curses.start_color()
                try:
                    curses.use_default_colors()
                except Exception:
                    pass
                try:
                    curses.init_pair(1, curses.COLOR_CYAN, -1)
                    curses.init_pair(2, curses.COLOR_GREEN, -1)
                    curses.init_pair(3, curses.COLOR_YELLOW, -1)
                    curses.init_pair(4, curses.COLOR_MAGENTA, -1)
                    curses.init_pair(5, curses.COLOR_RED, -1)
                    curses.init_pair(6, curses.COLOR_BLUE, -1)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            ascii_only_env = not _supports_unicode()
            self.cfg.ascii_only = bool(self.cfg.ascii_only or ascii_only_env)
        except Exception:
            self.cfg.ascii_only = True

        while True:
            self._consume_updates()
            self._render(stdscr)

            try:
                ch = stdscr.getch()
            except Exception:
                ch = -1

            if ch in (ord("p"), ord("P")):
                self.paused = not self.paused
            elif ch in (ord("d"), ord("D")):
                self.detached = True
                break
            elif ch in (ord("v"), ord("V")):
                self.view_idx = 1 - int(getattr(self, "view_idx", 0) or 0)
            elif ch in (ord("w"), ord("W")):
                self.window_idx = (self.window_idx + 1) % len(self.window_modes_ms)
            elif ch in (ord("c"), ord("C")):
                self.show_contact_markers = not self.show_contact_markers
            elif ch in (ord("n"), ord("N")):
                self.energy_normalize = not self.energy_normalize
            elif ch in (ord("s"), ord("S")):
                pth = self._save_screenshot(stdscr)
                if pth is None:
                    self.toast_msg = "save failed"
                else:
                    self.toast_msg = f"saved {pth.name}"
                self.toast_until = time.perf_counter() + 2.0
            elif ch in (ord("q"), ord("Q")):
                break

            if self.done_event.is_set() and not self.paused and not self.cfg.hold_on_done:
                break


def run_htop_monitor(
    updates: "queue.Queue[Dict[str, Any]]",
    done_event,
    title: str,
    refresh_s: float = 0.05,
    hold_on_done: bool = True,
    output_dir: Optional[Path] = None,
) -> None:
    """Run a full screen curses monitor."""
    cfg = MonitorConfig(refresh_s=refresh_s, hold_on_done=hold_on_done)
    mon = HtopLikeMonitor(
        updates=updates,
        done_event=done_event,
        title=title,
        cfg=cfg,
        output_dir=output_dir,
    )
    mon.run()
