"""Terminal live monitor.

This module provides a curses based live monitor inspired by htop.
It is designed to be light, dependency free, and usable over SSH
including mobile clients.

The engine can call a progress callback that pushes dict snapshots.
The CLI consumes those snapshots and renders a full screen monitor.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional, Iterable, Tuple, List

try:
    import curses  # type: ignore
except Exception:  # pragma: no cover
    curses = None  # type: ignore
import queue
import time


_SPARK = "▁▂▃▄▅▆▇█"
_SPARK_ASCII = " .:-=+*#%@"


def _supports_unicode() -> bool:
    try:
        if curses is None:
            return False
        return bool(curses.has_unicode_support())
    except Exception:
        return False


def _fmt_num(x: float) -> str:
    ax = abs(float(x))
    if ax >= 1e6:
        return f"{x:.3e}"
    if ax >= 1e3:
        return f"{x:.3g}"
    if ax >= 1.0:
        return f"{x:.3g}"
    if ax >= 1e-3:
        return f"{x:.3g}"
    return f"{x:.3e}"


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
            # fall back to nearest sample
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
        # Keep one column margin to avoid bottom right corner issues on some terminals
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
    ) -> None:
        self.updates = updates
        self.done_event = done_event
        self.title = title
        self.cfg = cfg or MonitorConfig()

        self.t0_wall = time.perf_counter()
        self.paused = False
        self.detached = False

        self.last: Dict[str, Any] = {}

        self.buf_t_ms: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.buf_force: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_pen: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_ratio: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_emech: Deque[float] = deque(maxlen=self.cfg.buffer_len)


        self.buf_ekin: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_ediss: Deque[float] = deque(maxlen=self.cfg.buffer_len)
        self.buf_enum: Deque[float] = deque(maxlen=self.cfg.buffer_len)

        self.peak_ekin = 0.0
        self.peak_ediss = 0.0
        self.peak_enum = 0.0

        self.peak_force = 0.0
        self.peak_pen = 0.0
        self.peak_ratio = 0.0

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
        em = float(latest.get("E_mech_J", 0.0) or 0.0)
        ek = float(latest.get("E_kin_J", 0.0) or 0.0)
        ed = float(latest.get("E_diss_total_J", 0.0) or 0.0)
        en = float(latest.get("E_num_J", 0.0) or 0.0)
        t_ms = float(latest.get("t", 0.0) or 0.0) * 1000.0

        self.buf_t_ms.append(t_ms)

        self.buf_force.append(f)
        self.buf_pen.append(p)
        self.buf_ratio.append(r)
        self.buf_emech.append(em)

        self.buf_ekin.append(ek)
        self.buf_ediss.append(ed)
        self.buf_enum.append(en)

        self.peak_force = max(self.peak_force, f)
        self.peak_pen = max(self.peak_pen, p)
        self.peak_ratio = max(self.peak_ratio, r)

        self.peak_ekin = max(self.peak_ekin, ek)
        self.peak_ediss = max(self.peak_ediss, ed)
        self.peak_enum = max(self.peak_enum, abs(en))


    def _draw_plot_core(
        self,
        win,
        top: int,
        left: int,
        height: int,
        width: int,
        x_ms: Deque[float],
        y_vals: Deque[float],
        y_unit: str,
        color_pair: int,
        downsample_kind: str = "max",
        show_y_labels: bool = True,
    ) -> None:
        if curses is None:
            return
        if height < 3 or width < 10:
            return

        xs = list(x_ms)
        ys = list(y_vals)
        if not xs or not ys:
            return

        ylab_w = 9 if show_y_labels else 0
        plot_h = max(1, height)
        plot_w = max(1, width - ylab_w)

        y_min = float(min(ys))
        y_max = float(max(ys))
        if abs(y_max - y_min) < 1e-30:
            y_max = y_min + 1.0
        else:
            span = y_max - y_min
            y_min = y_min - 0.05 * span
            y_max = y_max + 0.05 * span

        if downsample_kind == "last":
            xs_ds, ys_ds = _downsample_last(xs, ys, plot_w)
        else:
            xs_ds, ys_ds = _downsample_max(xs, ys, plot_w)

        if ylab_w > 0:
            for frac in (1.0, 0.5, 0.0):
                yy = y_min + frac * (y_max - y_min)
                row = int(round((1.0 - frac) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                lab = _fmt_num(yy).rjust(ylab_w - 1)
                _safe_addstr(win, top + row, left + 0, lab)

        ascii_only = bool(self.cfg.ascii_only)
        plot_char = "#" if ascii_only else "▇"
        line_char = "|" if ascii_only else "│"

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
            attr = 0
            try:
                if curses.has_colors() and color_pair > 0:
                    attr = curses.color_pair(color_pair)
            except Exception:
                attr = 0
            _safe_addstr(win, wy, wx, plot_char, attr)

            if prev_row is not None and prev_col is not None and col != prev_col:
                y0 = min(prev_row, row)
                y1 = max(prev_row, row)
                if y1 - y0 >= 2:
                    for rrow in range(y0 + 1, y1):
                        _safe_addstr(win, top + rrow, left + ylab_w + col, line_char, attr)
            prev_row = row
            prev_col = col

        _safe_addstr(win, top + plot_h - 1, left + ylab_w + 1, "ms")

    def _draw_plot_multi_core(
        self,
        win,
        top: int,
        left: int,
        height: int,
        width: int,
        x_ms: Deque[float],
        series: List[Tuple[str, Deque[float], int]],
        base_unit: str = "J",
    ) -> None:
        if curses is None:
            return
        if height < 3 or width < 10:
            return

        xs = list(x_ms)
        if not xs:
            return

        ys_list: List[List[float]] = []
        for _label, buf, _cp in series:
            ys_list.append(list(buf))
        if not ys_list or any(len(ys) != len(xs) for ys in ys_list):
            return

        abs_max = 0.0
        y_min = None
        y_max = None
        for ys in ys_list:
            if not ys:
                continue
            abs_max = max(abs_max, float(max(abs(v) for v in ys)))
            mn = float(min(ys))
            mx = float(max(ys))
            y_min = mn if y_min is None else min(y_min, mn)
            y_max = mx if y_max is None else max(y_max, mx)
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

        y_min_s = y_min / scale
        y_max_s = y_max / scale
        span = y_max_s - y_min_s
        y_min_s = y_min_s - 0.05 * span
        y_max_s = y_max_s + 0.05 * span
        if abs(y_max_s - y_min_s) < 1e-30:
            y_max_s = y_min_s + 1.0

        ylab_w = 9
        plot_h = max(1, height)
        plot_w = max(1, width - ylab_w)

        # Y labels
        for frac in (1.0, 0.5, 0.0):
            yy = y_min_s + frac * (y_max_s - y_min_s)
            row = int(round((1.0 - frac) * (plot_h - 1)))
            row = max(0, min(plot_h - 1, row))
            lab = _fmt_num(yy).rjust(ylab_w - 1)
            _safe_addstr(win, top + row, left + 0, lab)

        # Legend
        xoff = left + ylab_w + 1
        for label, _buf, cp in series:
            seg = f"{label} "
            attr = 0
            try:
                if curses.has_colors() and cp > 0:
                    attr = curses.color_pair(cp)
            except Exception:
                attr = 0
            _safe_addstr(win, top, xoff, seg, attr)
            xoff += len(seg)
            if xoff >= left + width - 2:
                break
        unit_txt = f"{unit}"
        _safe_addstr(win, top, left + width - len(unit_txt) - 1, unit_txt)

        ascii_only = bool(self.cfg.ascii_only)
        chars = ["*", "+", "x", "o", "#", "%"] if ascii_only else ["●", "◆", "▲", "■", "✚", "✖"]
        line_char = "|" if ascii_only else "│"

        for s_idx, (_label, _buf, cp) in enumerate(series):
            ys = ys_list[s_idx]
            _, ys_ds = _downsample_last(xs, ys, plot_w)
            ys_ds = [v / scale for v in ys_ds]

            plot_char = chars[s_idx % len(chars)]
            prev_row = None
            prev_col = None
            for j in range(len(ys_ds)):
                col = j
                yv = float(ys_ds[j])
                rr = (yv - y_min_s) / (y_max_s - y_min_s)
                rr = max(0.0, min(1.0, rr))
                row = int(round((1.0 - rr) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))

                wy = top + row
                wx = left + ylab_w + col
                attr = 0
                try:
                    if curses.has_colors() and cp > 0:
                        attr = curses.color_pair(cp)
                except Exception:
                    attr = 0
                _safe_addstr(win, wy, wx, plot_char, attr)

                if prev_row is not None and prev_col is not None and col != prev_col:
                    y0 = min(prev_row, row)
                    y1 = max(prev_row, row)
                    if y1 - y0 >= 2:
                        for rrow in range(y0 + 1, y1):
                            _safe_addstr(win, top + rrow, left + ylab_w + col, line_char, attr)

                prev_row = row
                prev_col = col

        _safe_addstr(win, top + plot_h - 1, left + ylab_w + 1, "ms")

    def _draw_box_plot(
        self,
        stdscr,
        top: int,
        left: int,
        height: int,
        width: int,
        title: str,
        x_ms: Deque[float],
        y_vals: Deque[float],
        last_val: float,
        peak_val: float,
        y_unit: str,
        color_pair: int = 0,
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

        ascii_only = bool(self.cfg.ascii_only)
        dot = "*" if ascii_only else "•"
        line = "|" if ascii_only else "│"

        attr_plot = 0
        try:
            if curses is not None and color_pair:
                attr_plot = curses.color_pair(int(color_pair))
        except Exception:
            attr_plot = 0

        inner_h = height - 2
        inner_w = width - 2

        ylab_w = 0
        if inner_w >= 52:
            ylab_w = 8

        plot_w = max(1, inner_w - ylab_w)
        plot_h = max(1, inner_h)

        # Title on top border
        header = f" {title}  last { _fmt_num(last_val) } {y_unit}  peak { _fmt_num(peak_val) } {y_unit} "
        header = header[: max(0, width - 2)]
        _safe_addstr(win, 0, 1, header)

        xs = list(x_ms)
        ys = list(y_vals)
        if len(xs) != len(ys):
            n = min(len(xs), len(ys))
            xs = xs[-n:]
            ys = ys[-n:]
        if len(xs) < 2:
            return

        x0 = float(xs[0])
        x1 = float(xs[-1])
        if x1 <= x0:
            x1 = x0 + 1.0

        y_min = 0.0
        y_max = max(ys) if len(ys) else 0.0
        if abs(y_max - y_min) < 1e-30:
            y_max = y_min + 1.0
        else:
            y_max = y_max * 1.05

        xs_ds, ys_ds = _downsample_max(xs, ys, plot_w)

        # Y axis labels
        if ylab_w > 0:
            for frac in (1.0, 0.5, 0.0):
                yy = y_min + frac * (y_max - y_min)
                row = int(round((1.0 - frac) * (plot_h - 1)))
                row = max(0, min(plot_h - 1, row))
                lab = _fmt_num(yy).rjust(ylab_w - 1)
                _safe_addstr(win, 1 + row, 1, lab)

        # Plot
        prev_row = None
        prev_col = None
        for j in range(len(ys_ds)):
            col = j
            yv = float(ys_ds[j])
            r = (yv - y_min) / (y_max - y_min)
            r = max(0.0, min(1.0, r))
            row = int(round((1.0 - r) * (plot_h - 1)))
            row = max(0, min(plot_h - 1, row))

            wy = 1 + row
            wx = 1 + ylab_w + col
            try:
                win.addstr(wy, wx, dot, attr_plot)
            except Exception:
                pass

            if prev_row is not None and prev_col is not None:
                if col == prev_col + 1 and row != prev_row:
                    a = min(row, prev_row)
                    b = max(row, prev_row)
                    for rr in range(a, b + 1):
                        wy2 = 1 + rr
                        wx2 = 1 + ylab_w + col
                        try:
                            win.addstr(wy2, wx2, line, attr_plot)
                        except Exception:
                            pass
            prev_row = row
            prev_col = col

        # X axis labels on bottom border
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
        x_ms: Deque[float],
        ratio_vals: Deque[float],
        last_ratio: float,
        peak_ratio: float,
        ek_vals: Deque[float],
        ed_vals: Deque[float],
        en_vals: Deque[float],
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
        stats = f"Eb last {last_ratio:.2e}  peak {peak_ratio:.2e}"
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

        # Ratio plot on top
        self._draw_plot_core(
            win,
            top=1,
            left=1,
            height=ratio_h,
            width=width - 2,
            x_ms=x_ms,
            y_vals=ratio_vals,
            y_unit="ratio",
            color_pair=3,
            downsample_kind="max",
            show_y_labels=True,
        )

        if ener_h < 3:
            return

        # Energies plot on bottom
        self._draw_plot_multi_core(
            win,
            top=sep_y + 1,
            left=1,
            height=ener_h,
            width=width - 2,
            x_ms=x_ms,
            series=[
                ("Ekin", ek_vals, 1),
                ("Ediss", ed_vals, 2),
                ("Enum", en_vals, 4),
            ],
            base_unit="J",
        )

    def _render(self, stdscr) -> None:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        wall = time.perf_counter() - self.t0_wall

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

        # --------------------------------------------------
        # Plots area
        # --------------------------------------------------
        y_top = 4
        footer_h = 1
        avail_h = max(0, h - footer_h - y_top)
        gap = 1
        n_pan = 3

        # If the terminal is tiny, fall back to one line sparklines.
        if avail_h < (n_pan * 5 + (n_pan - 1) * gap) or w < 60:
            y = 5
            plot_w = max(10, w - 28)

            def plot_row(label: str, buf, last_val: float, peak_val: float, y_row: int) -> None:
                spark = _sparkline(buf, plot_w, self.cfg.ascii_only)
                left = f"{label:<10}"
                right = f" last {last_val:9.3g}  peak {peak_val:9.3g}"
                _safe_addstr(stdscr, y_row, 0, left)
                _safe_addstr(stdscr, y_row, 11, spark)
                _safe_addstr(stdscr, y_row, 11 + plot_w + 1, right)

            if y < h:
                plot_row("force MN", self.buf_force, f, self.peak_force, y)
            if y + 1 < h:
                plot_row("pen mm", self.buf_pen, p, self.peak_pen, y + 1)
            if y + 2 < h:
                plot_row("Eb ratio", self.buf_ratio, ratio, self.peak_ratio, y + 2)
            if y + 3 < h:
                plot_row("E kin J", self.buf_ekin, float(self.last.get("E_kin_J", 0.0) or 0.0), self.peak_ekin, y + 3)
            if y + 4 < h:
                plot_row("E diss J", self.buf_ediss, float(self.last.get("E_diss_total_J", 0.0) or 0.0), self.peak_ediss, y + 4)
            if y + 5 < h:
                plot_row("E num J", self.buf_enum, float(self.last.get("E_num_J", 0.0) or 0.0), self.peak_enum, y + 5)
        else:
            pan_h = int((avail_h - (n_pan - 1) * gap) / n_pan)
            if pan_h < 5:
                pan_h = 5
            y0 = y_top
            self._draw_box_plot(
                stdscr,
                top=y0,
                left=0,
                height=pan_h,
                width=w,
                title="impact force",
                x_ms=self.buf_t_ms,
                y_vals=self.buf_force,
                last_val=f,
                peak_val=self.peak_force,
                y_unit="MN",
                color_pair=1,
            )
            y0 = y0 + pan_h + gap
            self._draw_box_plot(
                stdscr,
                top=y0,
                left=0,
                height=pan_h,
                width=w,
                title="indentation",
                x_ms=self.buf_t_ms,
                y_vals=self.buf_pen,
                last_val=p,
                peak_val=self.peak_pen,
                y_unit="mm",
                color_pair=2,
            )
            y0 = y0 + pan_h + gap
            self._draw_energy_panel(
                stdscr,
                top=y0,
                left=0,
                height=pan_h,
                width=w,
                x_ms=self.buf_t_ms,
                ratio_vals=self.buf_ratio,
                last_ratio=ratio,
                peak_ratio=self.peak_ratio,
                ek_vals=self.buf_ekin,
                ed_vals=self.buf_ediss,
                en_vals=self.buf_enum,
            )

        footer = "keys  q quit view  d detach  p pause"
        if self.done_event.is_set() and not self.cfg.hold_on_done:
            footer = "done  closing view"
        elif self.done_event.is_set() and self.cfg.hold_on_done:
            footer = "done  q quit view"
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
) -> None:
    """Run a full screen curses monitor.

    If curses is not available or the session is not a TTY, callers should
    fall back to simple line logging.
    """
    cfg = MonitorConfig(refresh_s=refresh_s, hold_on_done=hold_on_done)
    mon = HtopLikeMonitor(updates=updates, done_event=done_event, title=title, cfg=cfg)
    mon.run()
