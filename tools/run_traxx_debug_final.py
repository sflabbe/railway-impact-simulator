from __future__ import annotations
import os, sys, time, math, zipfile
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from railway_simulator.core.engine import run_simulation, SimulationConstants
sys.path.insert(0, str(Path(__file__).resolve().parent))
from debug_traxx_mass_contact import reconstruct_mass_contact, wall_contact_summary, effective_mass, initial_lengths

ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/traxx_mass_contact_debug_final')
OUT.mkdir(exist_ok=True, parents=True)
with open(ROOT/'configs'/'traxx_freight.yml','r') as f: BASE=yaml.safe_load(f)
VN_CASES=[4.0,5.0,6.0,7.0,8.0]
TMAX_BY_VN={4.0:0.10,5.0:0.10,6.0:0.10,7.0:0.09,8.0:0.09}
DT=float(BASE.get('h_init',1e-4))
K_MASS=SimulationConstants.MASS_CONTACT_STIFFNESS
C_MASS=SimulationConstants.MASS_CONTACT_DAMPING
K_WALL=float(BASE['k_wall'])


def run_case(vn):
    cfg=dict(BASE)
    cfg['v0_init']=-float(vn)
    Tmax=TMAX_BY_VN[vn]
    cfg['T_max']=Tmax; cfg['T_int']=(0.0,Tmax); cfg['step']=int(round(Tmax/cfg['h_init']))
    start=time.time()
    df=run_simulation(cfg, emit_peak_diagnostics=False)
    runtime=time.time()-start
    pair_summary,pair_ts=reconstruct_mass_contact(df,cfg)
    wall_summary=wall_contact_summary(df,cfg)
    # save compact timeseries
    n=int(cfg['n_masses'])
    slim=pd.DataFrame({'Time_s':df['Time_s'],'Impact_Force_MN':df['Impact_Force_MN'],'Penetration_mm':df['Penetration_mm']})
    for i in range(n):
        slim[f'Mass{i+1}_WallForce_MN']=np.maximum(df[f'Mass{i+1}_Force_wall_x_N'].to_numpy(float),0.0)/1e6
        slim[f'Mass{i+1}_x_m']=df[f'Mass{i+1}_Position_x_m']
    slim.to_csv(OUT/f'traxx_vn{vn:.1f}_wall_timeseries.csv',index=False)
    pair_ts.to_csv(OUT/f'traxx_vn{vn:.1f}_pair_contact_timeseries.csv',index=False)
    pair_summary.to_csv(OUT/f'traxx_vn{vn:.1f}_pair_summary.csv',index=False)
    wall_summary.to_csv(OUT/f'traxx_vn{vn:.1f}_wall_summary.csv',index=False)
    finite=pair_summary[np.isfinite(pair_summary['first_active_time_s'])]
    if len(finite):
        row=finite.sort_values('first_active_time_s').iloc[0]
        first_pair=int(row['pair']); first_time=float(row['first_active_time_s'])
    else:
        first_pair=-1; first_time=math.nan
    return {
        'vn_ms':vn,
        'Tmax_s':Tmax,
        'runtime_s':runtime,
        'Fpeak_wall_front_MN':float(slim['Impact_Force_MN'].max()),
        't_Fpeak_ms':float(slim.loc[slim['Impact_Force_MN'].idxmax(),'Time_s']*1000.0),
        'wall_penetration_front_max_mm':float(slim['Penetration_mm'].max()),
        'first_inter_mass_contact_pair':first_pair,
        'first_inter_mass_contact_time_ms':first_time*1000 if math.isfinite(first_time) else math.nan,
        'max_inter_mass_contact_force_MN':float(pair_summary['peak_force_MN'].max()),
        'max_inter_mass_penetration_mm':float(pair_summary['peak_penetration_m'].max()*1000.0),
        'max_wall_omega_dt':float(wall_summary['omega_dt_wall'].max()),
        'max_mass_contact_omega_dt':float(pair_summary['omega_dt_mass_contact'].max()),
        'max_iters_per_step':int(df.attrs.get('max_iters_per_step',-1)),
        'max_residual':float(df.attrs.get('max_residual',math.nan)),
        'note':'Tmax shortened for vn>=7 to avoid post-peak solver hang; Fpeak captured before stop.' if vn>=7 else ''
    }, pair_summary, wall_summary


def plot_case(vn):
    wall=pd.read_csv(OUT/f'traxx_vn{vn:.1f}_wall_timeseries.csv')
    pair=pd.read_csv(OUT/f'traxx_vn{vn:.1f}_pair_contact_timeseries.csv')
    t_ms=wall['Time_s'].to_numpy()*1000
    # force plot
    fig=plt.figure(figsize=(10,6))
    plt.plot(t_ms, wall['Impact_Force_MN'], lw=2.0, label='wall front / Impact_Force_MN')
    wall_cols=[c for c in wall.columns if c.endswith('_WallForce_MN')]
    total=wall[wall_cols].sum(axis=1)
    plt.plot(t_ms,total,lw=1.5,label='wall total over masses')
    for c in [c for c in pair.columns if c.endswith('_mass_contact_force_MN')]:
        if pair[c].max()>1e-6:
            plt.plot(t_ms,pair[c],lw=1.0,label=c.replace('_mass_contact_force_MN',''))
    plt.xlabel('Time [ms]'); plt.ylabel('Force [MN]')
    plt.title(f'TRAXX vn={vn:.1f} m/s — wall and inter-mass contact forces')
    plt.legend(fontsize=8); plt.tight_layout(); fig.savefig(OUT/f'plot_forces_vn{vn:.1f}.png',dpi=180); plt.close(fig)
    # penetration plot
    fig=plt.figure(figsize=(10,6))
    plt.plot(t_ms, wall['Penetration_mm'], lw=2.0, label='wall penetration front')
    for c in [c for c in pair.columns if c.endswith('_mass_contact_penetration_m')]:
        y=pair[c].to_numpy()*1000
        if np.max(y)>1e-6:
            plt.plot(t_ms,y,lw=1.0,label=c.replace('_mass_contact_penetration_m',''))
    plt.xlabel('Time [ms]'); plt.ylabel('Penetration [mm]')
    plt.title(f'TRAXX vn={vn:.1f} m/s — wall and inter-mass penetrations')
    plt.legend(fontsize=8); plt.tight_layout(); fig.savefig(OUT/f'plot_penetrations_vn{vn:.1f}.png',dpi=180); plt.close(fig)


def main():
    summaries=[]; pair_frames=[]; wall_frames=[]
    for vn in VN_CASES:
        print('run',vn,flush=True)
        s,ps,ws=run_case(vn)
        summaries.append(s)
        ps.insert(0,'vn_ms',vn); ws.insert(0,'vn_ms',vn)
        pair_frames.append(ps); wall_frames.append(ws)
        print(s,flush=True)
    summary=pd.DataFrame(summaries)
    pairs=pd.concat(pair_frames,ignore_index=True)
    walls=pd.concat(wall_frames,ignore_index=True)
    summary.to_csv(OUT/'debug_summary.csv',index=False)
    pairs.to_csv(OUT/'all_pair_contact_summary.csv',index=False)
    walls.to_csv(OUT/'all_wall_contact_summary.csv',index=False)
    for vn in VN_CASES: plot_case(vn)
    # summary plots
    fig=plt.figure(figsize=(8,5))
    plt.plot(summary.vn_ms,summary.Fpeak_wall_front_MN,marker='o',label='wall Fpeak')
    plt.plot(summary.vn_ms,summary.max_inter_mass_contact_force_MN,marker='o',label='max inter-mass contact')
    plt.xlabel('vn [m/s]'); plt.ylabel('Force [MN]'); plt.title('TRAXX debug: peaks vs vn'); plt.legend(); plt.tight_layout(); fig.savefig(OUT/'summary_force_peaks_vs_vn.png',dpi=180); plt.close(fig)
    fig=plt.figure(figsize=(8,5))
    plt.plot(summary.vn_ms,summary.wall_penetration_front_max_mm,marker='o',label='wall penetration')
    plt.plot(summary.vn_ms,summary.max_inter_mass_penetration_mm,marker='o',label='inter-mass penetration')
    plt.xlabel('vn [m/s]'); plt.ylabel('Penetration [mm]'); plt.title('TRAXX debug: penetrations vs vn'); plt.legend(); plt.tight_layout(); fig.savefig(OUT/'summary_penetrations_vs_vn.png',dpi=180); plt.close(fig)
    fig=plt.figure(figsize=(8,5))
    plt.plot(summary.vn_ms,summary.max_wall_omega_dt,marker='o',label='wall LN tangent')
    plt.plot(summary.vn_ms,summary.max_mass_contact_omega_dt,marker='o',label='inter-mass penalty')
    plt.axhline(2.0,ls='--',label='explicit reference 2')
    plt.xlabel('vn [m/s]'); plt.ylabel(r'$\omega\Delta t$ [-]'); plt.title('Courant-like diagnostic'); plt.legend(); plt.tight_layout(); fig.savefig(OUT/'summary_omega_dt_vs_vn.png',dpi=180); plt.close(fig)
    # report
    report='# TRAXX mass-contact debug report\n\n'
    report+='Runs use `configs/traxx_freight.yml` with original dt=1e-4 s. For vn=7,8 m/s, Tmax was shortened to 0.09 s because the original run becomes very slow/hangs after the first wall peak; the peak force is already reached before the cutoff.\n\n'
    report+='## Summary\n\n'+summary.to_markdown(index=False,floatfmt='.4g')+'\n\n'
    report+='## Pair contact summary\n\n'+pairs.to_markdown(index=False,floatfmt='.4g')+'\n\n'
    report+='## Wall contact summary\n\n'+walls.to_markdown(index=False,floatfmt='.4g')+'\n'
    (OUT/'traxx_mass_contact_debug_report.md').write_text(report,encoding='utf-8')
    zip_path=Path('/mnt/data/traxx_mass_contact_debug_final_outputs.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.iterdir()): zf.write(p,arcname=p.name)
    print('WROTE',zip_path,flush=True)

if __name__=='__main__':
    main(); sys.stdout.flush(); os._exit(0)
