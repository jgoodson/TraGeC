import importlib
import pkgutil
import sys
from pathlib import Path

from tragec.models.modeling import GeCModel, GeCConfig
from . import datasets

__version__ = '0.2'

for _, model, _ in pkgutil.iter_modules([str(Path(__file__).parent / 'models')]):
    imported_module = importlib.import_module('.models.' + model, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
                (issubclass(cls, GeCModel) or issubclass(cls, GeCConfig)):
            setattr(sys.modules[__name__], name, cls)
