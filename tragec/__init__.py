import importlib
import pkgutil
import sys
from pathlib import Path

from tragec.models.configuration import GeCConfig
from tragec.models.modeling import GeCModel
from . import datasets
from . import metrics

__version__ = '0.1'

# Import all the models and configs
for _, model, _ in pkgutil.iter_modules([str(Path(__file__).parent / 'models')]):
    imported_module = importlib.import_module('.models.' + model, package=__name__)
    for name, cls in imported_module.__dict__.items():
        if isinstance(cls, type) and \
                (issubclass(cls, GeCModel) or issubclass(cls, GeCConfig)):
            setattr(sys.modules[__name__], name, cls)
