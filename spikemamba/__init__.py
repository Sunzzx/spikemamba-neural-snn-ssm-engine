"""SpikeMamba: Event-Driven Neuromorphic Spiking State Space Model (SNN-SSM) Engine."""

__version__ = "0.1.0"

from .ssm import SelectiveSSM, ContinuousSSMConfig, associative_scan
from .neurons import LIFNeuron, AdaptiveLIFNeuron, SurrogateGradient, FastSigmoidSurrogate
from .model import SpikeMambaBlock, SpikeMambaModel
from .encoding import PoissonEncoder, RateDecoder, temporal_contrast_encoder

__all__ = [
    "SelectiveSSM",
    "ContinuousSSMConfig",
    "associative_scan",
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "SurrogateGradient",
    "FastSigmoidSurrogate",
    "SpikeMambaBlock",
    "SpikeMambaModel",
    "PoissonEncoder",
    "RateDecoder",
    "temporal_contrast_encoder",
]
