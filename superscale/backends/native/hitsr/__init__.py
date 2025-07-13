"""HiT-SR backend wrapper."""

# Import architectures from basicsr.archs
from .basicsr.archs.hit_sir_arch import HiT_SIR
from .basicsr.archs.hit_sng_arch import HiT_SNG  
from .basicsr.archs.hit_srf_arch import HiT_SRF

__all__ = ['HiT_SIR', 'HiT_SNG', 'HiT_SRF']
