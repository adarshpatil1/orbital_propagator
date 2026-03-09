import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import streamlit as st

# Constants
GM = 3.986e14
R_earth = 6.371e6
J2 = 1.08263e-3
Re = 6.371e6

def two_body(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    az = -GM * z / r**3
    return [vx, vy, vz, ax, ay, az]

def two_body_j2(t, state):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    az = -GM * z / r**3
    factor = (3/2) * J2 * GM * Re**2 / r**5
    ax += factor * x * (5*z**2/r**2 - 1)
    ay += factor * y * (5*z**2/r**2 - 1)
    az += factor * z * (5*z**2/r**2 - 3)
    return [vx, vy, vz, ax, ay, az]

# ── PAGE CONFIG ──
st.set_page_config(page_title="Orbital Propagator", layout="wide")
st.title("Two-Body Orbital Propagator")
st.markdown("Simulate satellite orbits from Newton's law of gravitation with optional J2 perturbation.")

# ── SIDEBAR INPUTS ──
st.sidebar.header("Orbit Parameters")
altitude = st.sidebar.slider("Altitude (km)", 200, 40000, 400, step=100)
inclination = st.sidebar.slider("Inclination (degrees)", 0, 90, 0, step=1)
num_orbits = st.sidebar.slider("Number of Orbits", 1, 20, 1)
enable_j2 = st.sidebar.checkbox("Enable J2 Perturbation", value=False)

# ── COMPUTE ──
r0 = R_earth + altitude * 1e3
v0 = np.sqrt(GM / r0)
T = 2 * np.pi * np.sqrt(r0**3 / GM)
T_total = num_orbits * T

inc_rad = np.radians(inclination)
state0 = [r0, 0, 0, 0, v0 * np.cos(inc_rad), v0 * np.sin(inc_rad)]

func = two_body_j2 if enable_j2 else two_body

with st.spinner("Simulating orbit..."):
    sol = solve_ivp(func, t_span=(0, T_total), y0=state0,
                    method='RK45', max_step=30, rtol=1e-9, atol=1e-9)

x, y, z = sol.y[0], sol.y[1], sol.y[2]
vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]

# Energy
r_vals = np.sqrt(x**2 + y**2 + z**2)
v_vals = np.sqrt(vx**2 + vy**2 + vz**2)
energy = 0.5 * v_vals**2 - GM / r_vals
energy_drift = abs(energy[-1] - energy[0])

# ── METRICS ──
st.subheader("Orbital Parameters")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Altitude", f"{altitude} km")
col2.metric("Orbital Speed", f"{v0/1000:.2f} km/s")
col3.metric("Period", f"{T/60:.1f} min")
col4.metric("Energy Drift", f"{energy_drift:.4f} J/kg")

# ── PLOT ──
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xe = R_earth * np.cos(u) * np.sin(v)
ye = R_earth * np.sin(u) * np.sin(v)
ze = R_earth * np.cos(v)
ax.plot_surface(xe, ye, ze, color='deepskyblue', alpha=0.4)
ax.plot(x, y, z, color='red', linewidth=1.2,
        label=f'{"J2 Perturbed" if enable_j2 else "Two-Body"} Orbit')

ax.set_title(f'Satellite Orbit — {altitude} km, {inclination}° inclination, {num_orbits} orbit(s)')
ax.legend()
max_range = r0 * 1.2
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
ax.set_box_aspect([1,1,1])
plt.tight_layout()
st.pyplot(fig)

# ── FOOTER NOTE ──
st.markdown("---")
st.markdown(
    "**Physics:** Newton's law of gravitation | "
    "**Integrator:** RK45 (SciPy) | "
    "**Validation:** Energy conservation"
)