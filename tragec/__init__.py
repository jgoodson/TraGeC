import importlib
import pkgutil
import sys
from pathlib import Path

from tragec.models.modeling import BioModel, BioConfig
from . import datasets

__version__ = '0.2'

for _, model, _ in pkgutil.iter_modules([str(Path(__file__).parent / 'models')]):
    imported_module = importlib.import_module('.models.' + model, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
                (issubclass(cls, BioModel) or issubclass(cls, BioConfig)):
            setattr(sys.modules[__name__], name, cls)
