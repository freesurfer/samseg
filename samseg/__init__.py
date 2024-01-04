import os
import sys
SAMSEGDIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

from .SamsegUtility import *
from .utilities import Specification, icv
from .Affine import Affine
from .GMM import GMM
from .GMM_fat_shift import GMM_fat_shift
from .Fatshift import Fatshift
from .BiasField import BiasField
from .ProbabilisticAtlas import ProbabilisticAtlas
from .Samseg import Samseg
from .SamsegLongitudinal import SamsegLongitudinal
# from .SamsegLesion import SamsegLesion
# from .SamsegLongitudinalLesion import SamsegLongitudinalLesion
from .figures import initVisualizer

from . import _version
__version__ = _version.get_versions()['version']
