from __future__ import annotations

import contextlib
import io
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from railway_simulator.core.engine import run_simulation, SimulationConstants

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/traxx_mass_contact_debug')
OUT.mkdir(parents=True, exist_ok=True)

VN_GRID = [4.0, 5.0, 6.0, 7.0, 8.0]
CONFIG_PATH = ROOT / 'configs' / 'traxx_freight.yml'

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    BASE = yaml.safe_load(f)

K_MASS = SimulationConstants.MASS_CONTACT_STIFFNESS
C_MASS = SimulationConstants.MASS_CONTACT_DAMPING
L_MIN_FRAC = SimulationConstants.MIN_SPRING_LENGTH_FRACTION
K_WALL = float(BASE['k_wall'])
DT = float(BASE.get('h_init', 1e-4))


def effective_mass(m1: float, m2: float) -> float:
    return m1 * m2 / (m1 + m2)


def initial_lengths(cfg: dict) -> np.ndarray:
    x = np.asarray(cfg['x_init'], dtype=float)
    y = np.asarray(cfg['y_init'], dtype=float)
    return np.hypot(np.diff(x), np.diff(y))


def reconstruct_mass_contact(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    masses = np.asarray(cfg['masses'], dtype=float)
    n = int(cfg['n_masses'])
    u10 = initial_lengths(cfg)
    lmin = L_MIN_FRAC * u10
    time_s = df['Time_s'].to_numpy(float)
    rows = []
    series = {'Time_s': time_s}

    for i in range(n-1):
        pair = i + 1
        x1 = df[f'Mass{i+1}_Position_x_m'].to_numpy(float)
        y1 = df[f'Mass{i+1}_Position_y_m'].to_numpy(float)
        x2 = df[f'Mass{i+2}_Position_x_m'].to_numpy(float)
        y2 = df[f'Mass{i+2}_Position_y_m'].to_numpy(float)
        vx1 = df[f'Mass{i+1}_Velocity_x_m_s'].to_numpy(float)
        vy1 = df[f'Mass{i+1}_Velocity_y_m_s'].to_numpy(float)
        vx2 = df[f'Mass{i+2}_Velocity_x_m_s'].to_numpy(float)
        vy2 = df[f'Mass{i+2}_Velocity_y_m_s'].to_numpy(float)
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        penetration = np.maximum(lmin[i] - L, 0.0)
        nx = np.divide(dx, L, out=np.ones_like(dx), where=L > 1e-12)
        ny = np.divide(dy, L, out=np.zeros_like(dy), where=L > 1e-12)
        vrel = (vx2-vx1)*nx + (vy2-vy1)*ny
        damping = np.where(vrel < 0.0, C_MASS * np.abs(vrel), 0.0)
        force = np.where(penetration > 0.0, K_MASS * penetration + damping, 0.0)
        active = penetration > 0.0
        if np.any(active):
            first_idx = int(np.argmax(active))
            first_time = float(time_s[first_idx])
        else:
            first_idx = -1
            first_time = math.nan
        peak_idx = int(np.argmax(force)) if force.size else -1
        rows.append({
            'pair': pair,
            'm_i_kg': masses[i],
            'm_j_kg': masses[i+1],
            'm_eff_kg': effective_mass(masses[i], masses[i+1]),
            'u10_m': u10[i],
            'L_min_m': lmin[i],
            'first_active_step': first_idx,
            'first_active_time_s': first_time,
            'peak_penetration_m': float(np.max(penetration)),
            'peak_force_N': float(np.max(force)),
            'peak_force_MN': float(np.max(force)/1e6),
            'omega_mass_contact_rad_s': math.sqrt(K_MASS / effective_mass(masses[i], masses[i+1])),
            'omega_dt_mass_contact': math.sqrt(K_MASS / effective_mass(masses[i], masses[i+1])) * DT,
        })
        series[f'Pair{pair}_mass_contact_penetration_m'] = penetration
        series[f'Pair{pair}_mass_contact_force_MN'] = force / 1e6
        series[f'Pair{pair}_mass_contact_active'] = active.astype(int)
        series[f'Pair{pair}_relative_normal_velocity_m_s'] = vrel
    return pd.DataFrame(rows), pd.DataFrame(series)


def wall_contact_summary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    masses = np.asarray(cfg['masses'], dtype=float)
    n = int(cfg['n_masses'])
    rows = []
    for i in range(n):
        x = df[f'Mass{i+1}_Position_x_m'].to_numpy(float)
        pen = np.maximum(-x, 0.0)
        fx = df[f'Mass{i+1}_Force_wall_x_N'].to_numpy(float)
        fpos = np.maximum(fx, 0.0)
        peak_pen = float(np.max(pen))
        kt = 1.5 * K_WALL * math.sqrt(peak_pen) if peak_pen > 0 else 0.0
        omega = math.sqrt(kt / masses[i]) if kt > 0 else 0.0
        rows.append({
            'mass': i+1,
            'mass_kg': masses[i],
            'peak_wall_penetration_m': peak_pen,
            'peak_wall_force_MN': float(np.max(fpos)/1e6),
            'wall_tangent_stiffness_N_m': kt,
            'omega_wall_rad_s': omega,
            'omega_dt_wall': omega * DT,
        })
    return pd.DataFrame(rows)


def make_plots(vn: float, df: pd.DataFrame, pair_series: pd.DataFrame, outdir: Path):
    t_ms = df['Time_s'].to_numpy(float) * 1000.0
    n = int(BASE['n_masses'])

    # wall + pair force overlaid
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t_ms, df['Impact_Force_MN'], linewidth=2.0, label='Wall contact: front mass / Impact_Force_MN')
    # total positive wall per all masses if any
    wall_total = np.zeros_like(t_ms)
    for i in range(n):
        wall_total += np.maximum(df[f'Mass{i+1}_Force_wall_x_N'].to_numpy(float), 0.0) / 1e6
    if np.max(wall_total) > 0:
        plt.plot(t_ms, wall_total, linewidth=1.5, label='Wall contact: sum over masses')
    for pair in range(1,n):
        y = pair_series[f'Pair{pair}_mass_contact_force_MN'].to_numpy(float)
        if np.max(y) > 1e-6:
            plt.plot(t_ms, y, linewidth=1.0, label=f'Inter-mass contact pair {pair}-{pair+1}')
    plt.xlabel('Time [ms]')
    plt.ylabel('Force [MN]')
    plt.title(f'TRAXX vn={vn:.1f} m/s: wall and inter-mass contact forces')
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / f'traxx_vn{vn:.1f}_forces.png', dpi=180)
    plt.close(fig)

    # penetrations
    fig = plt.figure(figsize=(10, 6))
    wall_pen = np.maximum(-df['Mass1_Position_x_m'].to_numpy(float), 0.0) * 1000.0
    plt.plot(t_ms, wall_pen, linewidth=2.0, label='Wall penetration mass 1')
    for pair in range(1,n):
        y = pair_series[f'Pair{pair}_mass_contact_penetration_m'].to_numpy(float) * 1000.0
        if np.max(y) > 1e-6:
            plt.plot(t_ms, y, linewidth=1.0, label=f'Inter-mass penetration pair {pair}-{pair+1}')
    plt.xlabel('Time [ms]')
    plt.ylabel('Penetration [mm]')
    plt.title(f'TRAXX vn={vn:.1f} m/s: wall and inter-mass penetrations')
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / f'traxx_vn{vn:.1f}_penetrations.png', dpi=180)
    plt.close(fig)

    # per-mass wall force
    fig = plt.figure(figsize=(10, 6))
    for i in range(n):
        y = np.maximum(df[f'Mass{i+1}_Force_wall_x_N'].to_numpy(float),0.0)/1e6
        if np.max(y)>1e-6:
            plt.plot(t_ms, y, linewidth=1.2, label=f'Mass {i+1}')
    plt.xlabel('Time [ms]')
    plt.ylabel('Wall contact force [MN]')
    plt.title(f'TRAXX vn={vn:.1f} m/s: wall contact by mass')
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(outdir / f'traxx_vn{vn:.1f}_wall_by_mass.png', dpi=180)
    plt.close(fig)


def run_case(vn: float):
    cfg = dict(BASE)
    cfg['v0_init'] = -float(vn)
    # Short horizon captures first wall peak and possible early inter-mass activation.
    cfg['T_max'] = 0.10
    cfg['T_int'] = (0.0, 0.10)
    cfg['step'] = int(round(0.10 / cfg.get('h_init', DT)))
    start = time.time()
    df = run_simulation(cfg, emit_peak_diagnostics=False)
    runtime = time.time()-start
    logtxt = ''
    print(f'    run finished vn={vn} runtime={runtime:.2f}s', flush=True)

    pair_summary, pair_series = reconstruct_mass_contact(df, cfg)
    print(f'    reconstruct finished vn={vn}', flush=True)
    wall_summary = wall_contact_summary(df, cfg)
    print(f'    wall summary finished vn={vn}', flush=True)
    # Plots generated in a separate lightweight pass to avoid backend shutdown issues.
    print(f'    plots skipped in run pass vn={vn}', flush=True)

    # First activation across all pairs
    active_times = pair_summary['first_active_time_s'].dropna()
    finite_times = active_times[np.isfinite(active_times)]
    if len(finite_times):
        first_time = float(finite_times.min())
        first_pair = int(pair_summary.loc[pair_summary['first_active_time_s'].idxmin(), 'pair'])
    else:
        first_time = math.nan
        first_pair = -1

    case_summary = {
        'vn_ms': vn,
        'runtime_s': runtime,
        'n_rows': int(len(df)),
        'dt_s': float(df.attrs.get('dt_eff', DT)),
        'Fpeak_wall_front_MN': float(df['Impact_Force_MN'].max()),
        'wall_penetration_front_max_mm': float(df['Penetration_mm'].max()),
        'first_inter_mass_contact_pair': first_pair,
        'first_inter_mass_contact_time_ms': first_time*1000.0 if math.isfinite(first_time) else math.nan,
        'max_inter_mass_contact_force_MN': float(pair_summary['peak_force_MN'].max()),
        'max_inter_mass_penetration_mm': float(pair_summary['peak_penetration_m'].max()*1000.0),
        'max_wall_omega_dt': float(wall_summary['omega_dt_wall'].max()),
        'max_mass_contact_omega_dt': float(pair_summary['omega_dt_mass_contact'].max()),
        'max_iters_per_step': int(df.attrs.get('max_iters_per_step', -1)),
        'max_residual': float(df.attrs.get('max_residual', math.nan)),
        'nonlinear_iters': int(df.attrs.get('n_nonlinear_iters', -1)),
        'log_excerpt': logtxt[-1000:],
    }
    print(f'    writing slim csvs vn={vn}', flush=True)
    # Store only debug-derived contact histories; raw full DataFrame is large and slow to serialize.
    slim_cols = ['Time_s','Impact_Force_MN','Penetration_mm','Velocity_m_s','Position_x_m']
    for c in df.columns:
        if ('Force_wall_x_N' in c) or ('Position_x_m' in c and c.startswith('Mass')):
            slim_cols.append(c)
    df[sorted(set(slim_cols), key=slim_cols.index)].to_csv(OUT / f'traxx_vn{vn:.1f}_slim_wall_timeseries.csv', index=False)
    pair_series.to_csv(OUT / f'traxx_vn{vn:.1f}_pair_contact_timeseries.csv', index=False)
    pair_summary.to_csv(OUT / f'traxx_vn{vn:.1f}_pair_summary.csv', index=False)
    wall_summary.to_csv(OUT / f'traxx_vn{vn:.1f}_wall_summary.csv', index=False)
    print(f'    run_case done vn={vn}', flush=True)
    return case_summary, pair_summary, wall_summary


def main():
    summaries=[]
    all_pair=[]
    all_wall=[]
    for vn in VN_GRID:
        print(f'Running vn={vn} m/s...', flush=True)
        cs, ps, ws = run_case(vn)
        summaries.append(cs)
        ps.insert(0,'vn_ms',vn)
        ws.insert(0,'vn_ms',vn)
        all_pair.append(ps)
        all_wall.append(ws)
        print('  Fpeak',cs['Fpeak_wall_front_MN'],'first MC pair',cs['first_inter_mass_contact_pair'],'time',cs['first_inter_mass_contact_time_ms'], flush=True)
    summary_df=pd.DataFrame(summaries)
    pair_df=pd.concat(all_pair,ignore_index=True)
    wall_df=pd.concat(all_wall,ignore_index=True)
    summary_df.to_csv(OUT/'debug_summary.csv',index=False)
    pair_df.to_csv(OUT/'all_pair_contact_summary.csv',index=False)
    wall_df.to_csv(OUT/'all_wall_contact_summary.csv',index=False)

    # summary plots
    fig=plt.figure(figsize=(8,5))
    plt.plot(summary_df['vn_ms'],summary_df['Fpeak_wall_front_MN'],marker='o',label='Fpeak wall front')
    plt.plot(summary_df['vn_ms'],summary_df['max_inter_mass_contact_force_MN'],marker='o',label='max inter-mass contact force')
    plt.xlabel('vn [m/s]')
    plt.ylabel('Force [MN]')
    plt.title('TRAXX debug summary: force peaks vs vn')
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUT/'summary_force_peaks_vs_vn.png',dpi=180)
    plt.close(fig)

    fig=plt.figure(figsize=(8,5))
    plt.plot(summary_df['vn_ms'],summary_df['wall_penetration_front_max_mm'],marker='o',label='wall penetration front')
    plt.plot(summary_df['vn_ms'],summary_df['max_inter_mass_penetration_mm'],marker='o',label='inter-mass penetration max')
    plt.xlabel('vn [m/s]')
    plt.ylabel('Penetration [mm]')
    plt.title('TRAXX debug summary: penetrations vs vn')
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUT/'summary_penetrations_vs_vn.png',dpi=180)
    plt.close(fig)

    fig=plt.figure(figsize=(8,5))
    plt.plot(summary_df['vn_ms'],summary_df['max_wall_omega_dt'],marker='o',label='wall LN tangent')
    plt.plot(summary_df['vn_ms'],summary_df['max_mass_contact_omega_dt'],marker='o',label='inter-mass penalty')
    plt.axhline(2.0,linestyle='--',label='explicit stability reference 2.0')
    plt.xlabel('vn [m/s]')
    plt.ylabel(r'$\omega\Delta t$ [-]')
    plt.title('Courant-like diagnostic')
    plt.legend()
    plt.tight_layout()
    fig.savefig(OUT/'summary_omega_dt_vs_vn.png',dpi=180)
    plt.close(fig)

    # Markdown report
    report = '# TRAXX mass-contact debug report\n\n'
    report += 'Config: `configs/traxx_freight.yml`; original dt/Tmax/settings preserved.\n\n'
    report += '## Summary\n\n'
    report += summary_df.drop(columns=['log_excerpt']).to_markdown(index=False, floatfmt='.4g')
    report += '\n\n## Pair contact summary\n\n'
    report += pair_df.to_markdown(index=False, floatfmt='.4g')
    report += '\n\n## Wall contact stiffness summary\n\n'
    report += wall_df.to_markdown(index=False, floatfmt='.4g')
    (OUT/'traxx_mass_contact_debug_report.md').write_text(report,encoding='utf-8')

    # zip
    import zipfile
    zip_path=Path('/mnt/data/traxx_mass_contact_debug_outputs.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.iterdir()):
            zf.write(p,arcname=p.name)
    print('Wrote',zip_path, flush=True)

if __name__=='__main__':
    main()
    import sys, os
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
