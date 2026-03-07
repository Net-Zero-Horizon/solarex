# SolareX - Solar Resource eXchange

[![Tests](https://github.com/msotocalvo/solarex/actions/workflows/tests.yml/badge.svg)](https://github.com/msotocalvo/solarex/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/1175040710.svg)](https://doi.org/10.5281/zenodo.18898422)

A Python library for solar PV resource assessment, irradiance analysis,
temperature modeling, shading/bifacial analysis, MCDA site scoring, and
capacity factor computation.

## Installation

```bash
pip install solarex
```

With optional dependencies:

```bash
pip install solarex[era5]    # ERA5 via atlite
pip install solarex[pvlib]   # CEC module database
pip install solarex[all]     # Everything
```

## Quick Start

```python
from solarex import (
    compute_peak_sun_hours,
    compute_gcr_shading_loss,
    compute_bifacial_gain,
    compute_pv_financials,
    SolarFinancialInputs,
)

# Financial analysis
inputs = SolarFinancialInputs(capacity_mw=50, capacity_factor=0.22)
result = compute_pv_financials(inputs)
print(f"LCOE: ${result.lcoe:.1f}/MWh, NPV: ${result.npv:,.0f}")

# Shading analysis
loss = compute_gcr_shading_loss(latitude=30, tilt=25, gcr=0.4)
print(f"Shading loss: {loss:.1%}")

# Bifacial gain
gain = compute_bifacial_gain(albedo=0.25, gcr=0.4, module_height=2.0, tilt=25)
print(f"Bifacial gain: {gain:.1%}")
```

## Modules

- `solarex.core` - Irradiance, temperature, shading, bifacial, capacity factor
- `solarex.data` - Open-Meteo, NASA POWER, ERA5, terrain, LULC, module DB
- `solarex.economics` - Financial analysis and LCOE sensitivity
- `solarex.analysis` - MCDA engine and grid evaluation
- `solarex.regional` - Regional analyzer and zone generation

## License

MIT
