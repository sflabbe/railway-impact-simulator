"""
Railway Impact Simulator package.

We keep this __init__ lightweight on purpose so that
`import railway_simulator` and `railway-sim --help` work
even if the heavy numerical engine is broken.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("railway-impact-simulator")
except PackageNotFoundError:  # during editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
