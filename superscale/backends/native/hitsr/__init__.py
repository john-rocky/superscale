"""HiT-SR backend wrapper."""

from .models.hit_arch import *
from .models.hit_sir_arch import *
from .models.hit_sng_arch import *
from .models.hit_srf_arch import *

# Simplified model creation
def create_model(opt):
    """Create HiT-SR model from options."""
    from .trainers.hit_sr_model import HiTSRModel
    return HiTSRModel(opt)
