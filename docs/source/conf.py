import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

project = "eml-pytorch"
copyright = "2026, UyNewNas"
author = "UyNewNas"

try:
    from eml_pytorch import __version__
    release = __version__
    version = ".".join(__version__.split(".")[:2])
except ImportError:
    release = "0.0.0.dev0"
    version = "0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

html_theme = "furo"
html_static_path = ["_static"]
