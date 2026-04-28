from .ops import eml
from .nn import EMLNode, TinyEMLNet, EMLActivation

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0.dev0"
    __version_tuple__ = (0, 0, 0, "dev0")
