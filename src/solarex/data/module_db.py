"""PV module database via pvlib CEC/NREL SAM.

Requires: pip install solarex[pvlib]
"""

from __future__ import annotations

import logging

from ..config import ModuleSpec

logger = logging.getLogger(__name__)


def load_module_database() -> list[ModuleSpec]:
    """Load CEC module database via pvlib.pvsystem.retrieve_sam('CECMod').

    Returns a list of ModuleSpec sorted by manufacturer then STC power.
    Typically ~21,535 modules from ~360 manufacturers.
    """
    import pvlib

    df = pvlib.pvsystem.retrieve_sam("CECMod")

    modules: list[ModuleSpec] = []
    for col_name in df.columns:
        try:
            mod = df[col_name]

            # Parse manufacturer from key (before double underscore)
            parts = col_name.split("__")
            manufacturer = parts[0].replace("_", " ") if len(parts) > 1 else "Unknown"

            # Parse display name (after double underscore)
            name = parts[1].replace("_", " ") if len(parts) > 1 else col_name.replace("_", " ")

            # Module area
            area = float(mod.get("A_c", 0) or 0)
            stc_w = float(mod.get("STC", 0) or 0)

            if stc_w <= 0:
                continue

            # Efficiency
            eff = stc_w / (area * 1000.0) if area > 0 else 0.0

            # Technology mapping
            tech_raw = str(mod.get("Technology", ""))
            tech = _normalize_technology(tech_raw)

            # Bifacial
            bifacial = bool(mod.get("Bifacial", 0))

            # Dimensions
            length = float(mod.get("Length", 0) or 0) / 1000.0  # mm -> m
            width = float(mod.get("Width", 0) or 0) / 1000.0

            modules.append(ModuleSpec(
                key=col_name,
                manufacturer=manufacturer,
                name=name,
                technology=tech,
                stc_power_w=stc_w,
                ptc_power_w=float(mod.get("PTC", 0) or 0),
                area_m2=area,
                efficiency=eff,
                bifacial=bifacial,
                v_oc=float(mod.get("V_oc_ref", 0) or 0),
                i_sc=float(mod.get("I_sc_ref", 0) or 0),
                v_mp=float(mod.get("V_mp_ref", 0) or 0),
                i_mp=float(mod.get("I_mp_ref", 0) or 0),
                gamma_pmax=float(mod.get("gamma_r", 0) or 0),
                t_noct=float(mod.get("T_NOCT", 45) or 45),
                n_cells=int(mod.get("N_s", 0) or 0),
                length_m=length,
                width_m=width,
            ))
        except Exception as exc:
            logger.debug("Skipping module %s: %s", col_name, exc)

    modules.sort(key=lambda m: (m.manufacturer.lower(), m.stc_power_w))
    return modules


def _normalize_technology(tech: str) -> str:
    """Normalize CEC technology string to readable form."""
    tech_lower = tech.lower().strip()
    if "mono" in tech_lower and "si" in tech_lower:
        return "Mono-c-Si"
    if "multi" in tech_lower and "si" in tech_lower:
        return "Multi-c-Si"
    if "cdte" in tech_lower or "cadmium" in tech_lower:
        return "CdTe"
    if "cigs" in tech_lower:
        return "CIGS"
    if "asi" in tech_lower or "amorphous" in tech_lower:
        return "a-Si"
    if "thin" in tech_lower:
        return "Thin Film"
    if tech:
        return tech
    return "Unknown"
