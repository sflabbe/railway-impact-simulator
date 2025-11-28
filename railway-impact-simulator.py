import streamlit as st
import numpy as np
from scipy.constants import g as GRAVITY
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ====================================================================
# A. HELPER FUNCTIONS
# ====================================================================

def dv_boucwen(x, u, A=1.0, beta=0.1, gamma=0.9, n=8, uy=0.05):
    """
    Bouc-Wen hysteretic evolution equation.
    x: current hysteretic state (dimensionless)
    u: velocity input
    """
    if abs(uy) < 1e-12:
        return 0.0
    return (A - np.abs(x)**n * (beta + np.sign(u * x) * gamma)) * u / uy

def rk_boucwen(x0, u, h, A=1.0, beta=0.1, gamma=0.9, n=8, uy=0.05):
    """
    4th-order Runge-Kutta integration for Bouc-Wen model.
    """
    k1 = dv_boucwen(x0, u, A, beta, gamma, n, uy)
    x_k2 = x0 + 0.5 * h * k1
    k2 = dv_boucwen(x_k2, u, A, beta, gamma, n, uy)
    x_k3 = x0 + 0.5 * h * k2
    k3 = dv_boucwen(x_k3, u, A, beta, gamma, n, uy)
    x_k4 = x0 + h * k3
    k4 = dv_boucwen(x_k4, u, A, beta, gamma, n, uy)
    return x0 + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def R_Vector_BoucWen(u, du, x, uy, fy, h, a=0.0,
                     A=1.0, beta=0.1, gamma=0.9, n=8):
    """
    u, du, x, uy, fy are 1D arrays of length n_springs.
    Returns:
        R  : nodal force vector of length n_springs+1
        xfunc : updated hysteretic state array (same shape as x)
    """
    u = np.asarray(u, dtype=float)
    du = np.asarray(du, dtype=float)
    x = np.asarray(x, dtype=float)
    uy = np.asarray(uy, dtype=float)
    fy = np.asarray(fy, dtype=float)

    n_springs = len(u)
    xfunc = np.zeros_like(x)
    f2 = np.zeros(n_springs + 1)
    R = np.zeros(n_springs + 1)

    for i in range(n_springs):
        if i == 0:
            # First spring → first node force
            xfunc[0] = rk_boucwen(x[0], du[0], h, A, beta, gamma, n)
            f2[0] = a * (fy[0] / uy[0]) * u[0] + (1.0 - a) * fy[0] * xfunc[0]
            R[0] = -f2[0]
        else:
            # Left spring contribution
            xfunc[i-1] = rk_boucwen(x[i-1], du[i-1], h, A, beta, gamma, n)
            f2[i-1] = a * (fy[i-1] / uy[i-1]) * u[i-1] + (1.0 - a) * fy[i-1] * xfunc[i-1]
            R_temp1 = f2[i-1]

            # Right spring contribution
            xfunc[i] = rk_boucwen(x[i], du[i], h, A, beta, gamma, n)
            f2[i] = a * (fy[i] / uy[i]) * u[i] + (1.0 - a) * fy[i] * xfunc[i]
            R_temp2 = f2[i]

            # Internal node force = left spring − right spring
            R[i] = R_temp1 - R_temp2

    # Last node (n+1)
    xfunc[n_springs-1] = rk_boucwen(x[n_springs-1], du[n_springs-1], h,
                                    A, beta, gamma, n)
    f2[n_springs-1] = (a * (fy[n_springs-1] / uy[n_springs-1]) * u[n_springs-1] +
                       (1.0 - a) * fy[n_springs-1] * xfunc[n_springs-1])
    R[n_springs] = f2[n_springs-1]

    return R, xfunc


def lugre_friction(z_prev, v, F_coulomb, F_static, v_stribeck, sigma_0, sigma_1, sigma_2, h):
    """LuGre friction model from lugref.m"""
    v_stribeck = max(abs(v_stribeck), 1e-10)
    r = -(v / v_stribeck)**2
    g = (F_coulomb + (F_static - F_coulomb) * np.exp(r)) / sigma_0
    z_dot = v - np.abs(v) * z_prev / g
    z = z_prev + z_dot * h
    F = sigma_0 * z + sigma_1 * z_dot + sigma_2 * v
    return F, z

def dahl_friction(z_prev, v, F_coulomb, sigma_0, h):
    """Dahl friction model from dahlf.m"""
    Fc = max(abs(F_coulomb), 1e-12)
    z_dot = (1.0 - sigma_0 / Fc * z_prev * np.sign(v)) * v
    z = z_prev + z_dot * h
    F = sigma_0 * z
    return F, z

def contact_force_ye_mod(u_kontakt, du_kontakt, v0kt, k_wall, cr_wall):
    """
    Ye et al. modified contact model.
    u_kontakt: penetration (negative when in contact)
    du_kontakt: penetration velocity
    v0kt: initial contact velocity
    """
    R = k_wall * u_kontakt * (1 + 3*(1 - cr_wall)/(2*cr_wall) * (du_kontakt / v0kt))
    R[R > 0] = 0  # Only compression
    return R

def to_excel(df):
    """Generate Excel file for download."""
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Dynamic Load History')
    except ImportError:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Dynamic Load History')
    return output.getvalue()

# ====================================================================
# B. MAIN SIMULATION ENGINE
# ====================================================================

def run_simulation(params):
    """
    Main HHT-alpha simulation
    """
    # Extract parameters
    n      = params['n_masses']
    mass   = params['masses']
    x_init = np.asarray(params['x_init'], dtype=float)
    y_init = np.asarray(params['y_init'], dtype=float)
    v0_init = params['v0_init']
    winkel  = params['angle_rad']

    # Material properties (arrays for each spring)
    fy = np.asarray(params['fy'], dtype=float)   # shape (n-1,)
    uy = np.asarray(params['uy'], dtype=float)   # shape (n-1,)

    # Per-spring stiffness k_i = fy_i / uy_i
    k_init = fy / uy                              # shape (n-1,)

    k_wall  = params['k_wall']
    cr_wall = params['cr_wall']

    # Friction parameters
    mu_s          = params['mu_s']
    mu_k          = params['mu_k']
    sigma_0       = params['sigma_0']
    sigma_1       = params['sigma_1']
    sigma_2       = params['sigma_2']
    friction_model = params['friction_model']

    # Bouc–Wen parameters
    bw_a     = params['bw_a']
    bw_A     = params['bw_A']
    bw_beta  = params['bw_beta']
    bw_gamma = params['bw_gamma']
    bw_n     = params['bw_n']

    # Time parameters
    step  = int(params['step'])
    T_int = params['T_int']
    h = (T_int[1] - T_int[0]) / step
    t = np.linspace(T_int[0], T_int[1], step + 1)

    # Rotation matrix for impact angle
    rot = np.array([[np.cos(winkel), -np.sin(winkel)],
                    [np.sin(winkel),  np.cos(winkel)]])

    # Rotate initial positions into global coordinates
    xy_init = rot @ np.vstack([x_init, y_init])
    x_init = xy_init[0, :]
    y_init = xy_init[1, :]

    # Initial velocities along train axis, then rotated
    xp_init = np.full(n, v0_init)
    yp_init = np.zeros(n)

    xpyp_init = rot @ np.vstack([xp_init, yp_init])
    xp_init = xpyp_init[0, :]
    yp_init = xpyp_init[1, :]

    # Mass matrix
    M0 = np.diag(np.concatenate([mass, mass]))

    # --- 2D bar-element stiffness matrix ---
    dof2 = 2 * n
    K0 = np.zeros((dof2, dof2))
    for i in range(n - 1):
        j = i + 1
    
        dx = x_init[j] - x_init[i]
        dy = y_init[j] - y_init[i]
        L0 = np.hypot(dx, dy)
        if L0 == 0:
            continue
    
        cx = dx / L0
        cy = dy / L0
        k = k_init[i]            # element stiffness
    
        ke = k * np.array([
            [ cx*cx,  cx*cy, -cx*cx, -cx*cy],
            [ cx*cy,  cy*cy, -cx*cy, -cy*cy],
            [-cx*cx, -cx*cy,  cx*cx,  cx*cy],
            [-cx*cy, -cy*cy,  cx*cy,  cy*cy]
        ])
    
        dofs = [i, n + i, j, n + j]  # x_i, y_i, x_j, y_j
        for a in range(4):
            for b in range(4):
                K0[dofs[a], dofs[b]] += ke[a, b]
    
    # --- Rayleigh damping (use highest two positive eigenfrequencies) ---
    eigenvalues, _ = np.linalg.eig(np.linalg.solve(M0, K0))
    positive = np.real(eigenvalues[np.real(eigenvalues) > 1e-12])
    freqs = np.sqrt(np.abs(positive))
    freqs = np.real(freqs)
    freqs.sort()
    zeta = 0.05
    
    if len(freqs) >= 2:
        wn1, wn2 = freqs[-2], freqs[-1]
        alpha_r = 2 * zeta * wn1 * wn2 / (wn1 + wn2)
        beta_r = 2 * zeta / (wn1 + wn2)
    elif len(freqs) == 1:
        wn1 = freqs[0]
        alpha_r = 2 * zeta * wn1
        beta_r = 0.0
    else:
        alpha_r = 0.0
        beta_r = 0.0
    
    C = M0 * alpha_r + K0 * beta_r
    
    # --- Initial conditions ---
    q0 = np.concatenate([x_init, y_init])
    qp0 = np.concatenate([xp_init, yp_init])
    
    # At the undeformed configuration u = 0 ⇒ Vq0 = 0
    Vq0 = np.zeros_like(q0)


    # State arrays
    xf   = np.zeros((2 * n, step + 1))
    xfp  = np.zeros((2 * n, step + 1))
    xfpp = np.zeros((2 * n, step + 1))

    xf[:, 0] = q0
    xfp[:, 0] = qp0
    xfpp[:, 0] = np.linalg.solve(M0, -Vq0)  # now simply zeros


    # HHT-alpha parameters
    alpha = 0.3
    beta  = (1.0 + alpha) ** 2 / 4.0
    gamma = 0.5 + alpha
    tol   = 1e-3

    dof = n

    # Initial spring lengths u10 (2D geometry)
    u10 = np.zeros(dof - 1)
    for i in range(dof - 1):
        r10 = np.array([x_init[i],     y_init[i]])
        r20 = np.array([x_init[i + 1], y_init[i + 1]])
        u10[i] = np.linalg.norm(r20 - r10)

    # Force and state tracking
    u1         = np.zeros((dof - 1, step + 1))
    u_kontakt  = np.zeros((2 * dof, step + 1))
    Rkontakt   = np.zeros((2 * dof, step + 1))
    Ukontakt   = np.zeros((2 * dof, step + 1))
    R_springPlot = np.zeros((2 * dof, step + 1))
    Freibung   = np.zeros((2 * dof, step + 1))

    # Bouc-Wen hysteretic states per spring
    Xfunc = np.zeros((dof - 1, step + 1))

    # Friction states (LuGre / Dahl)
    zlugre = np.zeros((2 * dof, step + 1))

    # Contact tracking
    blkt = 0
    v0kt = np.ones(2 * dof)

    # Normal forces for friction
    gpp = GRAVITY
    FN  = gpp * (M0 @ np.ones(2 * n))

    # Time-stepping
    for l in range(step):
        xfpp_pred = xfpp[:, l].copy()
        xfpp[:, l + 1] = xfpp_pred

        err = 1e3
        iter_count = 0
        max_iter = 50

        while err > tol and iter_count < max_iter:
            iter_count += 1

            # Predictor (HHT-alpha)
            xf[:, l + 1] = (xf[:, l] +
                            h * xfp[:, l] +
                            (0.5 - beta) * h ** 2 * xfpp[:, l] +
                            beta * h ** 2 * xfpp[:, l + 1])

            xfp[:, l + 1] = (xfp[:, l] +
                             (1.0 - gamma) * h * xfpp[:, l] +
                             gamma * h * xfpp[:, l + 1])

            # Spring deformations (2D geometry)
            for p in range(dof - 1):
                r1 = np.array([xf[p,         l + 1],
                               xf[dof + p,   l + 1]])
                r2 = np.array([xf[p + 1,     l + 1],
                               xf[dof + p+1, l + 1]])
                u1[p, l + 1] = np.linalg.norm(r2 - r1) - u10[p]

            # Spring deformation rate
            if l > 0:
                du = (u1[:, l + 1] - u1[:, l]) / h
            else:
                du = np.zeros(dof - 1)

            # Bouc–Wen internal forces
            R_spring2, Xfunc[:, l+1] = R_Vector_BoucWen(
                -u1[:, l+1], -du, Xfunc[:, l], uy, fy, h,
                a=bw_a, A=bw_A, beta=bw_beta, gamma=bw_gamma, n=bw_n
            )
            
            # Map spring nodal forces to full DOF vector (x-DOFs, y-DOFs=0)
            R_spring = np.concatenate([R_spring2, np.zeros(dof)])
            R_springPlot[:, l+1] = R_spring

            # Friction forces
            Fc_r = mu_k * np.abs(FN)
            Fs_r = mu_s * np.abs(FN)
            vs = 0.001

            Fr_gravel = np.zeros(2 * dof)
            for rk in range(2 * dof):
                if friction_model == 'dahl':
                    Fr_gravel[rk], zlugre[rk, l + 1] = dahl_friction(
                        zlugre[rk, l], xfp[rk, l + 1], Fc_r[rk], sigma_0, h
                    )
                else:
                    Fr_gravel[rk], zlugre[rk, l + 1] = lugre_friction(
                        zlugre[rk, l], xfp[rk, l], Fc_r[rk], Fs_r[rk], vs,
                        sigma_0, sigma_1, sigma_2, h
                    )

            # Contact forces at the wall (x < 0)
            R = np.zeros(2 * dof)
            if np.any(xf[:dof, l + 1] < 0.0):
                # contact deformation in x
                for p in range(dof):
                    if xf[p, l + 1] < 0.0:
                        r1 = np.array([0.0,              xf[dof + p, l + 1]])
                        r2 = np.array([xf[p, l + 1],      xf[dof + p, l + 1]])
                        u_kontakt[p, l + 1] = -np.linalg.norm(r2 - r1)

                if l > 0:
                    du_kontakt = (u_kontakt[:, l + 1] - u_kontakt[:, l]) / h
                else:
                    du_kontakt = np.zeros(2 * dof)

                # Track initial contact velocity
                if not blkt and np.any(du_kontakt[:dof] <= 0.0):
                    blkt = 1
                    v0kt = np.where(du_kontakt < 0.0, du_kontakt, 1.0)
                    v0kt[v0kt == 0.0] = 1.0

                # Ye et al. modified contact model
                R = contact_force_ye_mod(u_kontakt[:, l + 1], du_kontakt, v0kt, k_wall, cr_wall)

                Rkontakt[:, l + 1] = R
                Ukontakt[:, l + 1] = u_kontakt[:, l + 1]
            elif l > 0 and np.any(u_kontakt[:dof, l] < 0.0):
                blkt = 0
                v0kt = np.ones(2 * dof)

            # HHT-alpha acceleration update
            xfpp[:, l + 1] = np.linalg.solve(
                M0,
                ((1.0 - alpha) * R_spring + alpha * R_springPlot[:, l] - Vq0
                 - (1.0 - alpha) * Rkontakt[:, l + 1] - alpha * Rkontakt[:, l]
                 - (1.0 - alpha) * Fr_gravel - alpha * Freibung[:, l]
                 - (1.0 - alpha) * (C @ xfp[:, l + 1]) - alpha * (C @ xfp[:, l]))
            )

            # Convergence check
            err = np.linalg.norm(xfpp[:, l + 1] - xfpp_pred)
            xfpp_pred = xfpp[:, l + 1].copy()
            Freibung[:, l + 1] = Fr_gravel

    # Post-processing
    F_total       = -Rkontakt[0, :]         # front mass contact force
    u_penetration = -Ukontakt[0, :] * 1000  # mm
    a_front       = xfpp[0, :] / GRAVITY    # g

    export_df = pd.DataFrame({
        'Time_s': t,
        'Time_ms': t * 1000.0,
        'Impact_Force_MN': F_total / 1e6,
        'Penetration_mm': u_penetration,
        'Acceleration_g': a_front,
        'Velocity_m_s': xfp[0, :],
        'Position_x_m': xf[0, :],
        'BoucWen_State_1': Xfunc[0, :] if dof > 1 else np.zeros_like(t),
    })

    return export_df, F_total, u_penetration, a_front, xfpp[:dof, :], t

# ====================================================================
# C. STREAMLIT UI
# ====================================================================

st.set_page_config(layout="wide", page_title="Railway Impact Simulator")

st.title("🚂 Railway Impact Simulator")
st.markdown("**HHT-α Implicit Integration with Bouc-Wen Hysteresis**")

with st.sidebar:
    st.header("⚙️ Parameters")
    
    with st.expander("🕐 Time & Velocity", expanded=True):
        v0_kmh = st.slider("Impact Velocity (km/h)", 10, 200, 56, 1)
        v0_init = -v0_kmh / 3.6
        
        T_final = st.number_input("Simulation Time (s)", 0.01, 1.0, 0.3, 0.01)
        step = st.number_input("Time Steps", 1000, 200000, 100000, 1000)
        
        angle_deg = st.number_input("Impact Angle (°)", 0.0, 45.0, 0.0, 0.1)
        angle_rad = angle_deg * (np.pi/2) / 90
    
    with st.expander("🚃 Train Geometry", expanded=True):
        n_masses = st.slider("Number of Masses", 2, 20, 7)
        
        default_masses = np.array([4, 10, 4, 4, 4, 10, 4]) * 1000
        default_x = np.array([0.02, 3.02, 6.52, 10.02, 13.52, 17.02, 20.02])
        default_y = np.zeros(7)
        
        if n_masses == 7:
            masses = default_masses
            x_init = default_x
            y_init = default_y
        else:
            M_total = st.number_input("Total Mass (kg)", 100.0, 100000.0, 40000.0, 100.0)
            masses = np.ones(n_masses) * M_total / n_masses
            
            l_total = st.number_input("Total Length (m)", 1.0, 100.0, 20.0, 0.1)
            x_init = np.linspace(0, l_total, n_masses)
            y_init = np.zeros(n_masses)
    
    with st.expander("🔧 Bouc-Wen Material", expanded=True):
        fy_MN = st.number_input("Yield Force F_y (MN)", 0.1, 100.0, 15.0, 0.1)
        fy = np.ones(n_masses - 1) * fy_MN * 1e6
        
        uy_mm = st.number_input("Yield Deformation u_y (mm)", 1.0, 500.0, 200.0, 1.0)
        uy = np.ones(n_masses - 1) * uy_mm / 1000
        
        st.write(f"Stiffness: {(fy[0]/uy[0])/1e6:.1f} MN/m")
        
        bw_a = st.slider("Elastic ratio (a)", 0.0, 1.0, 0.0, 0.05)
        bw_A = st.number_input("A", 0.1, 10.0, 1.0, 0.1)
        bw_beta = st.number_input("β", 0.0, 5.0, 0.1, 0.05)
        bw_gamma = st.number_input("γ", 0.0, 5.0, 0.9, 0.05)
        bw_n = int(st.number_input("n", 1, 20, 8, 1))
    
    with st.expander("💥 Contact", expanded=True):
        k_wall = st.number_input("Wall Stiffness (MN/m)", 1.0, 100.0, 45.0, 1.0) * 1e6
        cr_wall = st.slider("Coeff. of Restitution", 0.1, 0.99, 0.8, 0.01)
    
    with st.expander("🛞 Friction", expanded=True):
        friction_model = st.selectbox("Model", ['lugre', 'dahl'])
        mu_s = st.slider("μ_s", 0.0, 1.0, 0.4, 0.01)
        mu_k = st.slider("μ_k", 0.0, 1.0, 0.3, 0.01)
        sigma_0 = st.number_input("σ_0", 1e3, 1e7, 1e5, format="%.0e")
        sigma_1 = st.number_input("σ_1", 1.0, 1e5, 316.0, 1.0)
        sigma_2 = st.number_input("σ_2", 0.0, 2.0, 0.4, 0.1)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Configuration")
    st.metric("Velocity", f"{v0_kmh} km/h")
    st.metric("Masses", n_masses)
    st.metric("Time Steps", step)
    st.markdown("---")
    run_btn = st.button("▶️ **Run Simulation**", type="primary", width="stretch")

with col2:
    if run_btn:
        params = {
            'n_masses': n_masses, 'masses': masses,
            'x_init': x_init, 'y_init': y_init,
            'v0_init': v0_init, 'angle_rad': angle_rad,
            'fy': fy, 'uy': uy,
            'k_wall': k_wall, 'cr_wall': cr_wall,
            'mu_s': mu_s, 'mu_k': mu_k,
            'sigma_0': sigma_0, 'sigma_1': sigma_1, 'sigma_2': sigma_2,
            'friction_model': friction_model,
            'bw_a': bw_a, 'bw_A': bw_A, 'bw_beta': bw_beta,
            'bw_gamma': bw_gamma, 'bw_n': bw_n,
            'step': step, 'T_int': [0, T_final]
        }
        
        with st.spinner("Running HHT-α simulation..."):
            try:
                export_df, F_total, u_pen, a_front, acc_field, t = run_simulation(params)
                st.success("✅ Complete!")
            except Exception as e:
                st.error(f"Error: {e}")
                raise
        
        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Force", f"{F_total.max()/1e6:.2f} MN")
        c2.metric("Max Penetration", f"{u_pen.max():.2f} mm")
        c3.metric("Max Acceleration", f"{a_front.max():.1f} g")
        
        # Plots
        fig = make_subplots(rows=4, cols=1,
                           subplot_titles=('Force', 'Penetration', 'Acceleration', 'Hysteresis'),
                           vertical_spacing=0.08)
        
        fig.add_trace(go.Scatter(x=export_df['Time_ms'], y=export_df['Impact_Force_MN'],
                                line=dict(color='red', width=2)), row=1, col=1)
        fig.update_yaxes(title_text="Force (MN)", row=1, col=1)
        
        fig.add_trace(go.Scatter(x=export_df['Time_ms'], y=export_df['Penetration_mm'],
                                line=dict(color='blue', width=2)), row=2, col=1)
        fig.update_yaxes(title_text="Penetration (mm)", row=2, col=1)
        
        fig.add_trace(go.Scatter(x=export_df['Time_ms'], y=export_df['Acceleration_g'],
                                line=dict(color='green', width=2)), row=3, col=1)
        fig.update_yaxes(title_text="Acceleration (g)", row=3, col=1)
        fig.update_xaxes(title_text="Time (ms)", row=3, col=1)
        
        fig.add_trace(go.Scatter(x=export_df['Penetration_mm'], y=export_df['Impact_Force_MN'],
                                mode='markers', marker=dict(color=export_df['Time_ms'],
                                colorscale='Viridis', size=3, colorbar=dict(title="Time (ms)"))),
                     row=4, col=1)
        fig.update_xaxes(title_text="Penetration (mm)", row=4, col=1)
        fig.update_yaxes(title_text="Force (MN)", row=4, col=1)
        
        fig.update_layout(height=1400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        
        # Export
        st.subheader("📥 Export")
        c1, c2, c3 = st.columns(3)
        c1.download_button("📄 CSV", export_df.to_csv(index=False).encode(), "results.csv", "text/csv", width="stretch")
        c2.download_button("📝 TXT", export_df.to_string(index=False).encode(), "results.txt", "text/plain", width="stretch")
        c3.download_button("📊 XLSX", to_excel(export_df), "results.xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
    else:
        st.info("👈 Configure and run simulation")