"""
UI package for the Railway Impact Simulator Streamlit application.

This package provides modular components for the web-based user interface,
organized by functionality for better maintainability and testing.
"""

from .about import display_citation, display_header
from .export import to_excel
from .parameters import build_parameter_ui
from .plotting import create_results_plots
from .sdof import (
    compute_building_sdof_response,
    compute_force_response_spectrum,
    compute_multi_damping_force_response_spectrum,
    create_building_animation,
    create_building_hysteresis_plot,
    create_building_response_plots,
    create_multi_damping_response_spectrum_plot,
    create_response_spectrum_plot,
)
from .simulation import execute_simulation
from .train_geometry import create_train_geometry_plot

__all__ = [
    # About page
    "display_header",
    "display_citation",
    # Export
    "to_excel",
    # Plotting
    "create_results_plots",
    # SDOF
    "compute_building_sdof_response",
    "compute_force_response_spectrum",
    "compute_multi_damping_force_response_spectrum",
    "create_building_animation",
    "create_building_hysteresis_plot",
    "create_building_response_plots",
    "create_multi_damping_response_spectrum_plot",
    "create_response_spectrum_plot",
    # Train geometry
    "create_train_geometry_plot",
    # Parameters
    "build_parameter_ui",
    # Simulation
    "execute_simulation",
]
