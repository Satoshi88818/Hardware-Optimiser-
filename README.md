# Hardware Optimiser v5

Abstract, domain-agnostic engineering hardware design optimiser.

## Package structure

```
hardware_optimiser/
├── __init__.py          Full public API re-export
├── __main__.py          python -m hardware_optimiser entry point
├── _compat.py           Optional-dependency flags (JAX, sklearn, pymoo, shm)
├── config.py            DomainConfig — JSON/YAML loading
├── dataclasses_.py      Frozen I/O dataclasses (NozzleParams, MotorResults, …)
├── registry.py          PhysicsRegistry — named function dispatch
├── interfaces.py        ABCs: AbstractEvaluator, HardwareDomain, OptimResult, …
├── objective.py         WeightedObjective, ObjectiveTerm
├── autodiff.py          JAX gradient provider, make_jax_objective
├── constraints.py       ConstraintKind/Spec/Set, build_constraint_set
├── optimizers.py        GenericOptimizer (SLSQP+JAX), FeasibilityRepair,
│                        ParallelSLSQPEvaluator, DifferentialEvolutionOptimizer,
│                        ParallelMultiStartOptimizer, NSGAParetoOptimizer,
│                        ParetoFront, ParetoSolution
├── surrogate.py         SurrogateEvaluator (GP/RBF), SurrogateStats,
│                        BayesianOptimizer, BOResult
├── sampling.py          LHS/Sobol samplers, MultiStartResults, run_multistart
├── tolerance.py         ToleranceMCResults, run_tolerance_mc,
│                        MassModelRegistry, SurrogateTolMCResults
├── factory.py           get_domain / get_optimizer / get_surrogate / …
├── runner.py            run_single / run_sampling / run_bayesopt / …
├── tests.py             17 unit tests (--test flag)
├── cli.py               argparse main()
├── dashboard.py         Streamlit dashboard launcher
└── domains/
    ├── __init__.py
    ├── nozzle.py          de Laval rocket nozzle  (JAX physics + OpenSCAD)
    ├── electric_motor.py  PMSM electric motor     (JAX physics + OpenSCAD)
    └── centrifugal_pump.py Centrifugal pump       (OpenSCAD + FreeCAD macro)
```

## Installation

```bash
pip install .                     # core (numpy, scipy, scikit-learn)
pip install ".[jax]"              # + JAX autodiff
pip install ".[pareto]"           # + NSGA-II/III (pymoo)
pip install ".[all]"              # everything
```

## CLI quick reference

```bash
# Single SLSQP run
python -m hardware_optimiser --domain nozzle

# Multi-start (LHS)
python -m hardware_optimiser --domain electric_motor --mode monte_carlo --n-runs 32

# Differential evolution
python -m hardware_optimiser --domain centrifugal_pump --mode differential_evo

# NSGA-II Pareto front
python -m hardware_optimiser --domain nozzle --mode nsga2 --pop-size 100 --n-gen 200

# Bayesian optimisation (GP-EI)
python -m hardware_optimiser --domain electric_motor --mode bayesopt --bo-budget 60

# Surrogate-based single run
python -m hardware_optimiser --domain nozzle --mode surrogate_single

# Tolerance Monte-Carlo
python -m hardware_optimiser --domain nozzle --tolerance-mc 2000

# Fast surrogate tolerance MC
python -m hardware_optimiser --domain nozzle --surrogate-tol-mc 5000

# Surrogate uncertainty at optimum
python -m hardware_optimiser --domain electric_motor --surrogate-stats

# CAD export (OpenSCAD / FreeCAD)
python -m hardware_optimiser --domain nozzle --export-cad openscad
python -m hardware_optimiser --domain centrifugal_pump --export-cad freecad

# Constraint classification table
python -m hardware_optimiser --domain centrifugal_pump --show-constraints

# Run all unit tests
python -m hardware_optimiser --test

# Launch Streamlit dashboard
python -m hardware_optimiser --dashboard
```

## Python API example

```python
from hardware_optimiser import get_domain, DomainConfig, get_surrogate, get_bo_optimizer

cfg    = DomainConfig(surrogate="gp", doe_size=24, bo_budget=50)
domain = get_domain("nozzle", cfg)

# Standard SLSQP
result = domain.optimizer.optimize_design(domain.effective_params())
print(result.scalar_results())

# Bayesian optimisation
surrogate = get_surrogate(domain, cfg)
bo        = get_bo_optimizer(domain, surrogate, cfg, seed=42)
bo_result = bo.run()
print(bo_result.report())

# Surrogate uncertainty at any point
stats = surrogate.stats_at(result.optimized_params)
print(stats.report())

# Constraint classification (overridable per-name in config)
cset = domain.constraint_set()
print(cset.summary())
```

## Config file (JSON / YAML)

```json
{
  "bounds":           { "throat_radius": [0.03, 0.08] },
  "weights":          { "specific_impulse": -2.0, "mass": 0.2 },
  "tols":             { "throat_radius": 0.005 },
  "constraint_kinds": { "isp_ge_min": "soft" },
  "doe_size":         32,
  "bo_budget":        60,
  "surrogate":        "gp"
}
```

Pass with `--config my_config.json`.
