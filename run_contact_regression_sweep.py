from pathlib import Path
import contextlib, io
import yaml
import pandas as pd
from railway_simulator.core.engine import run_simulation
from railway_simulator.hazard.sdof import equivalent_static_force_sdof
import numpy as np

root = Path(__file__).resolve().parent
with (root/'configs'/'traxx_freight.yml').open(encoding='utf-8') as f:
    base = yaml.safe_load(f)
models=['flores','anagnostopoulos','lankarani-nikravesh','hooke','hertz']
vns=[5.75,6.0,6.25]
rows=[]
for model in models:
    for vn in vns:
        cfg=dict(base)
        cfg.update(contact_model=model, v0_init=-vn, T_max=0.10, T_int=(0,0.10), step=1000, h_init=1e-4, solver='picard')
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            df=run_simulation(cfg, emit_peak_diagnostics=False)
        t=df['Time_s'].to_numpy(float); f=df['Impact_Force_MN'].to_numpy(float)
        extra=np.arange(t[-1]+0.001,1.5+0.0005,0.001)
        tpad=np.concatenate([t,extra]); fpad=np.concatenate([f,np.zeros_like(extra)])
        feq=equivalent_static_force_sdof(tpad,fpad,Tn_s=0.1,zeta=0.05,oscillator_mass=1.0)
        rows.append({'model':model,'vn_ms':vn,'Fpeak_MN':float(f.max()),'Feq_100ms_MN':float(feq),'DAF':float(feq/f.max()),'t_Fpeak_ms':float(df['Time_ms'].iloc[int(f.argmax())]),'penetration_max_mm':float(df['Penetration_mm'].max())})
        print(model, vn, f.max(), feq, flush=True)
out=root/'contact_regression_sweep_final.csv'
pd.DataFrame(rows).to_csv(out,index=False)
print(out)
