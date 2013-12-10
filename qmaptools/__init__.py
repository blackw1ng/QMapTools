"""
qmaptools - a library for quantifying elemental maps 

Copyright (c) 2013 Sebastian Werner <sebastian.werner@gmail.com>
See LICENSE file for details and (absence of) warranty

qmaptools: http://github.com/blackw1ng/qmaptools

"""

__VERSION__ = "0.1"
import numpy as np

# bring submodules into package scope
from .image_importer import *
from .correlation import *

# some basic definitions



# General settings
from warnings import simplefilter,filterwarnings
simplefilter("ignore", np.ComplexWarning)
filterwarnings("ignore")