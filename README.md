# Two-Body Orbital Propagator with J2 Perturbation

A satellite orbit simulator built from first principles using Python.

## What This Project Does
- Propagates satellite orbits using Newton's law of gravitation
- Validates simulation through mechanical energy conservation
- Models J2 oblateness perturbation (Earth's equatorial bulge effect)
- Demonstrates nodal precession over 15 orbits (~1 day)
- Compares LEO, MEO, and GEO orbital regimes

## Results
| Orbit | Altitude | Speed | Period |
|-------|----------|-------|--------|
| LEO (ISS) | 400 km | 7.67 km/s | 92.4 min |
| MEO (GPS) | 20,200 km | 3.87 km/s | 718.4 min |
| GEO | 35,786 km | 3.07 km/s | 1435.7 min |

## Tech Stack
Python, NumPy, SciPy, Matplotlib

## Key Concepts
- Newton's law of gravitation
- First-order ODE formulation
- RK45 numerical integration
- J2 perturbation theory
- Nodal precession