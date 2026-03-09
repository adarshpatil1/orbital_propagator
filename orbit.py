import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
GM = 3.986e14  # Earth's gravitational parameter (m^3/s^2)
R_earth = 6.371e6  # Earth radius in meters
J2 = 1.08263e-3 # j2 ratio
Re = 6.371e6  # Earth's equatorial radius

# --- Define the equations of motion ---
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
    
    # Pure gravity (same as before)
    ax = -GM * x / r**3
    ay = -GM * y / r**3
    az = -GM * z / r**3
    
    # J2 perturbation - extra acceleration from Earth's bulge
    factor = (3/2) * J2 * GM * Re**2 / r**5
    ax += factor * x * (5*z**2/r**2 - 1)
    ay += factor * y * (5*z**2/r**2 - 1)
    az += factor * z * (5*z**2/r**2 - 3)
    
    return [vx, vy, vz, ax, ay, az]

# --- Initial conditions (ISS-like orbit) ---
# Position: 400 km above Earth's surface
altitude = 400e3
r0 = R_earth + altitude       # ~6771 km from Earth center

# Circular orbit velocity
v0 = np.sqrt(GM / r0)         # ~7669 m/s

# Start at (r0, 0, 0), moving in y-direction
state0 = [r0, 0, 0, 0, v0, 0]

# --- Orbital period (so we simulate exactly one orbit) ---
T = 2 * np.pi * np.sqrt(r0**3 / GM)
print(f"Orbital period: {T/60:.1f} minutes")

# --- Integrate ---
sol = solve_ivp(
    two_body,
    t_span=(0, T),
    y0=state0,
    method='RK45',
    max_step=10,        # 10 second steps
    rtol=1e-9,
    atol=1e-9
)

x, y, z = sol.y[0], sol.y[1], sol.y[2]

# --- Plot ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw Earth
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xe = R_earth * np.cos(u) * np.sin(v)
ye = R_earth * np.sin(u) * np.sin(v)
ze = R_earth * np.cos(v)
ax.plot_surface(xe, ye, ze, color='deepskyblue', alpha=0.4)

# Draw orbit
ax.plot(x, y, z, color='red', linewidth=1.5, label='Satellite Orbit')

ax.set_title('Two-Body Orbital Propagation\n(ISS-like orbit, 400 km altitude)', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('orbit.png', dpi=150)
plt.show()

# --- Verify: check energy is conserved ---
r_vals = np.sqrt(x**2 + y**2 + z**2)
vx, vy, vz = sol.y[3], sol.y[4], sol.y[5]
v_vals = np.sqrt(vx**2 + vy**2 + vz**2)
energy = 0.5 * v_vals**2 - GM / r_vals
print(f"Energy at start: {energy[0]:.2f} J/kg")
print(f"Energy at end:   {energy[-1]:.2f} J/kg")
print(f"Energy drift:    {abs(energy[-1]-energy[0]):.4f} J/kg  ← should be tiny")


# Simulate 15 orbits
T_total = 15 * T

# Run both - pure two-body and J2
sol_pure = solve_ivp(two_body, t_span=(0, T_total), y0=state0,
                     method='RK45', max_step=10, rtol=1e-9, atol=1e-9)

# For J2 we need a tilted initial orbit to see precession
# Add slight inclination - z component to velocity
state0_inclined = [r0, 0, 0, 0, v0 * np.cos(np.radians(51.6)), v0 * np.sin(np.radians(51.6))]

sol_j2 = solve_ivp(two_body_j2, t_span=(0, T_total), y0=state0_inclined,
                   method='RK45', max_step=10, rtol=1e-9, atol=1e-9)

# Plot both
fig = plt.figure(figsize=(14, 6))

# Pure two-body
ax1 = fig.add_subplot(121, projection='3d')
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xe = R_earth*np.cos(u)*np.sin(v)
ye = R_earth*np.sin(u)*np.sin(v)
ze = R_earth*np.cos(v)
ax1.plot_surface(xe, ye, ze, color='deepskyblue', alpha=0.3)
ax1.plot(sol_pure.y[0], sol_pure.y[1], sol_pure.y[2], 
         color='green', linewidth=0.8)
ax1.set_title('Pure Two-Body\n(no perturbation)', fontsize=11)

# J2 perturbed
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(xe, ye, ze, color='deepskyblue', alpha=0.3)
ax2.plot(sol_j2.y[0], sol_j2.y[1], sol_j2.y[2], 
         color='red', linewidth=0.8)
ax2.set_title('J2 Perturbed Orbit\n(Earth oblateness effect)', fontsize=11)

plt.suptitle('Two-Body vs J2 Perturbation — 15 Orbits (ISS inclination 51.6°)', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('j2_comparison.png', dpi=150)
plt.show()

print(f"\nSimulating {15} orbits = {15*92.4:.0f} minutes = {15*92.4/60:.1f} hours")
print("J2 orbit uses ISS inclination of 51.6 degrees")
print("Notice how the J2 orbit slowly rotates — that is nodal precession")


# ── LEO / MEO / GEO COMPARISON ──────────────────

altitudes = {
    'LEO (ISS - 400 km)':    400e3,
    'MEO (GPS - 20,200 km)': 20200e3,
    'GEO (36,000 km)':       35786e3,
}

colors_list = ['red', 'orange', 'cyan']

fig2 = plt.figure(figsize=(10, 8))
ax3 = fig2.add_subplot(111, projection='3d')

# Draw Earth
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xe2 = R_earth * np.cos(u) * np.sin(v)
ye2 = R_earth * np.sin(u) * np.sin(v)
ze2 = R_earth * np.cos(v)
ax3.plot_surface(xe2, ye2, ze2, color='deepskyblue', alpha=0.3)

print("\n── Orbit Comparison ──────────────────────────")
print(f"{'Orbit':<25} {'Altitude':>12} {'Speed':>12} {'Period':>14}")
print("-" * 65)

for (label, alt), color in zip(altitudes.items(), colors_list):
    r = R_earth + alt
    v_orb = np.sqrt(GM / r)
    period = 2 * np.pi * np.sqrt(r**3 / GM)

    s0 = [r, 0, 0, 0, v_orb, 0]

    sol = solve_ivp(two_body, t_span=(0, period), y0=s0,
                    method='RK45', max_step=60, rtol=1e-9, atol=1e-9)

    ax3.plot(sol.y[0], sol.y[1], sol.y[2],
             color=color, linewidth=1.5, label=label)

    print(f"{label:<25} {alt/1000:>8.0f} km  {v_orb/1000:>8.2f} km/s  {period/60:>10.1f} min")

ax3.set_title('LEO vs MEO vs GEO — One Orbit Each', fontsize=13, fontweight='bold')
ax3.legend(loc='upper left')
plt.tight_layout()
plt.savefig('leo_meo_geo.png', dpi=150)
plt.show()