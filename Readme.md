
# Hardware Optimiser v5

**Abstract, domain-agnostic engineering design optimization framework**  
with JAX automatic differentiation, Gaussian Process / RBF surrogates, Bayesian Optimization, uncertainty-aware reporting, and manufacturing tolerance Monte-Carlo.

Current version: **v5** (March 2026)

## ✨ Key Features (v5 highlights)

- **JAX-powered gradients** — full autodiff replaces hand-written derivatives  
- **Domain-owned constraint classification** — default kinds (`hard_ineq`, `soft`, `hard_eq`) overridable via config  
- **Gaussian Process & RBF surrogates** — fast predictions + uncertainty (μ, σ, P05/P95, worst-case)  
- **Bayesian Optimization (GP + Expected Improvement)** — active learning loop with configurable budget  
- **Surrogate-aware tolerance Monte-Carlo** — thousands of cheap samples for realistic robustness assessment  
- **Multi-objective Pareto fronts** via NSGA-II / NSGA-III (pymoo)  
- **Feasibility projection repair** — projects infeasible SLSQP solutions back to feasible region  
- **CAD export hooks** — OpenSCAD (nozzle, motor) + FreeCAD macro (pump)  
- **Streamlit dashboard** — visual configuration & live results (run with `--dashboard`)

## Supported Domains (2026)

| Domain             | Physics model          | CAD output          | Typical goals                              |
|---------------------|------------------------|---------------------|--------------------------------------------|
| `nozzle`           | 1D isentropic flow     | OpenSCAD            | Maximize Isp / thrust, respect area ratio, length |
| `electric_motor`   | Surface-mount PMSM     | OpenSCAD            | Maximize power/torque density, respect thermal & saturation limits |
| `centrifugal_pump` | 1D mean-line + slip    | OpenSCAD + FreeCAD  | Target head & flow @ max efficiency, min power, acceptable NPSHr |

Extensible — adding new domains requires ~300–600 LOC (physics + constraints + geometry).

## Installation

```bash
# Minimal (most features)
pip install numpy scipy jax jaxlib matplotlib pyyaml streamlit

# For Gaussian Process surrogates
pip install scikit-learn

# For NSGA-II / NSGA-III Pareto fronts
pip install pymoo

# Optional — faster parallel multi-start
pip install tqdm  # just visual feedback
```

JAX is configured to use **CPU double precision** by default (`jax_enable_x64=True`).  
GPU support is possible but currently disabled in the code.

## Quick Start

```bash
# Single run — fastest way to see results
python hardware_optimiser_v5.py --domain nozzle --mode single

# Bayesian Optimization (recommended sweet spot)
python hardware_optimiser_v5.py --domain electric_motor --mode bayesopt --bo-budget 60

# Pareto front (multi-objective)
python hardware_optimiser_v5.py --domain centrifugal_pump --mode nsga2 --pop-size 80 --n-gen 150

# Surrogate-only optimization (very fast after DoE)
python hardware_optimiser_v5.py --domain nozzle --mode surrogate_single

# Tolerance robustness study on best design (physics)
python hardware_optimiser_v5.py --domain electric_motor --mode single --tolerance-mc 2000

# ... same but using cheap GP surrogate (much faster)
python hardware_optimiser_v5.py --domain electric_motor --mode single --surrogate-tol-mc 10000

# Launch interactive dashboard
python hardware_optimiser_v5.py --dashboard
```

## Configuration (JSON or YAML)

```yaml
# example: config_pump.yaml
params:
  target_head_m: 55.0
  flow_rate_m3s: 0.06

bounds:
  impeller_diameter_m: [0.18, 0.45]
  blade_angle_deg:     [18.0, 42.0]

weights:
  hydraulic_efficiency: -2.5
  shaft_power_W:         1.2

constraint_kinds:
  head_ge_min:         hard_ineq
  npsh_r_le_max:       soft
  tip_speed_le_max:    soft

doe_size:  32
bo_budget: 70
surrogate: gp       # or "rbf"
```

Run with: `--config config_pump.yaml`

## Current Limitations (March 2026)

| # | Limitation                                      | Severity | Workaround / Comment                                                                 |
|---|--------------------------------------------------|----------|---------------------------------------------------------------------------------------|
| 1 | Single-file monolith (~1500–2000 LOC active)     | High     | Hard to maintain & test; planned split into package                                   |
| 2 | Bayesian Optimization is sequential              | High     | No parallel / batch acquisition → slow on expensive physics                           |
| 3 | Only basic GP (sklearn) & RBF (scipy)            | Medium   | No deep kernels, TuRBO-style local trust-regions, multi-fidelity, heteroscedastic GP |
| 4 | Soft constraints use simple quadratic penalty    | Medium   | No augmented Lagrangian / exact l₁ penalty; can cause oscillation or slow progress   |
| 5 | No constraint-tightening / annealing schedule    | Medium   | Important for problems with many tight constraints                                    |
| 6 | Tolerance MC only on scalar parameters           | Medium   | No spatial/temporal fields or correlated manufacturing errors                        |
| 7 | Limited visualization outside Streamlit dashboard| Medium   | No built-in convergence plots, parallel coordinate plots, etc.                        |
| 8 | JAX forced to CPU + double precision             | Low–Med  | Easy to enable GPU; current choice is for reproducibility                             |
| 9 | No multi-fidelity / variable-fidelity support    | Low–Med  | Common in real aerospace/turbomachinery workflows                                     |

## Planned / Desired Future Enhancements

Priority order (2026–2027 horizon)

1. Refactor into proper Python package (`src/hardware_optimiser/`, `pyproject.toml`, tests)
2. Parallel / batched candidate evaluation in Bayesian Optimization loop
3. Support for more powerful BO libraries (`botorch`, `GPyTorch`, `Ax`, `TuRBO`-style local trust regions)
4. Better soft-constraint handling (augmented Lagrangian, adaptive penalty schedule)
5. Multi-fidelity / hierarchical evaluation support
6. Built-in visualization suite (convergence, Pareto front, sensitivity, surrogate slices)
7. Correlated manufacturing error models in tolerance MC
8. GPU acceleration toggle for JAX
9. More domains (axial compressor stage, heat exchanger, battery pack thermal, small satellite structure, …)
10. Export to common formats (STEP/IGES via CadQuery / Open CASCADE, JSON for external solvers)

Contributions welcome — especially on items 1–4.

## Philosophy & Design Axioms

1. Physics first — everything traceable to first-principles equations
2. Gradients should be free — JAX autodiff eliminates most derivative bugs
3. Uncertainty is not optional — report mean, std, quantiles, worst-case at every optimum
4. Constraints carry intent — domain experts declare meaning, product managers override strictness
5. Speed matters — surrogates + cheap MC make iteration times acceptable for real projects

## License

MIT (assuming open-source intent — change if needed)

## Acknowledgments

Built with:  
JAX ⋅ SciPy ⋅ scikit-learn ⋅ pymoo ⋅ Streamlit ⋅ numpy ⋅ yaml

Happy optimizing!

— Notso / Serious 
(March 2026)
```
