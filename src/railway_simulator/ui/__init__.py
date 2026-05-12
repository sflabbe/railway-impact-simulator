"""UI package for the Railway Impact Simulator Streamlit application.

The package intentionally uses lazy attribute imports.  Some modules require the
optional ``streamlit`` dependency, while lightweight helpers such as SRS plotting
are useful in tests and non-UI contexts.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    # About page
    "display_header": (".about", "display_header"),
    "display_citation": (".about", "display_citation"),
    # Export
    "to_excel": (".export", "to_excel"),
    # Plotting
    "create_results_plots": (".plotting", "create_results_plots"),
    # SDOF
    "compute_building_sdof_response": (".sdof", "compute_building_sdof_response"),
    "compute_force_response_spectrum": (".sdof", "compute_force_response_spectrum"),
    "compute_multi_damping_force_response_spectrum": (".sdof", "compute_multi_damping_force_response_spectrum"),
    "create_building_animation": (".sdof", "create_building_animation"),
    "create_building_hysteresis_plot": (".sdof", "create_building_hysteresis_plot"),
    "create_building_response_plots": (".sdof", "create_building_response_plots"),
    "create_multi_damping_response_spectrum_plot": (".sdof", "create_multi_damping_response_spectrum_plot"),
    "create_response_spectrum_plot": (".sdof", "create_response_spectrum_plot"),
    # Train geometry
    "create_train_geometry_plot": (".train_geometry", "create_train_geometry_plot"),
    # Parameters
    "build_parameter_ui": (".parameters", "build_parameter_ui"),
    # Simulation
    "execute_simulation": (".simulation", "execute_simulation"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
