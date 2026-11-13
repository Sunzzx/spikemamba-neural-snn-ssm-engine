"""Neuromorphic Event Encoding and Decoding Primitives."""
from __future__ import annotations

from typing import Tuple
import numpy as np


class PoissonEncoder:
    """Converts real-valued continuous tensors into stochastic spike trains via Poisson point process."""

    def __init__(self, time_steps: int, max_rate_hz: float = 100.0, dt_s: float = 0.001, seed: int = 42):
        self.time_steps = time_steps
        self.max_rate_hz = max_rate_hz
        self.dt_s = dt_s
        self.rng = np.random.default_rng(seed)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode continuous input x in [0, 1] to binary spikes (batch, time_steps, dim)."""
        x_norm = np.clip(x, 0.0, 1.0)
        p_spike = x_norm * self.max_rate_hz * self.dt_s  # Probability of spike per step
        p_spike = np.clip(p_spike, 0.0, 1.0)
        
        # Broadcast across time_steps
        shape = (x.shape[0], self.time_steps, x.shape[1]) if x.ndim == 2 else (1, self.time_steps, x.shape[0])
        p_expanded = np.repeat(p_spike[:, None, :], self.time_steps, axis=1) if x.ndim == 2 else np.repeat(p_spike[None, None, :], self.time_steps, axis=1)
        
        rand_uniform = self.rng.uniform(0.0, 1.0, size=shape)
        return (rand_uniform < p_expanded).astype(np.float64)


class RateDecoder:
    """Decodes binary spike trains into continuous predictions via mean firing rate or exponential trace."""

    @staticmethod
    def decode_mean_rate(spikes: np.ndarray) -> np.ndarray:
        """Compute average spike count across time dimension.
        Args:
            spikes: (batch, seq_len, dim)
        Returns:
            rate: (batch, dim)
        """
        return np.mean(spikes, axis=1)

    @staticmethod
    def decode_exponential_trace(spikes: np.ndarray, decay: float = 0.9) -> np.ndarray:
        """Filter spikes using causal exponential low-pass filter.
        Args:
            spikes: (batch, seq_len, dim)
        Returns:
            filtered: (batch, seq_len, dim)
        """
        batch, seq_len, dim = spikes.shape
        filtered = np.zeros_like(spikes)
        trace = np.zeros((batch, dim), dtype=np.float64)
        for t in range(seq_len):
            trace = decay * trace + (1.0 - decay) * spikes[:, t, :]
            filtered[:, t, :] = trace
        return filtered


def temporal_contrast_encoder(signal: np.ndarray, threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Dynamic Vision Sensor (DVS) style temporal contrast encoder.
    Generates ON and OFF event polarity streams from derivative thresholding.
    
    Args:
        signal: (batch, seq_len, dim) continuous analog signal
        threshold: Delta threshold to trigger positive or negative spike
    Returns:
        on_spikes: (batch, seq_len, dim) positive polarity events
        off_spikes: (batch, seq_len, dim) negative polarity events
    """
    diff = np.diff(signal, axis=1, prepend=signal[:, :1, :])
    on_spikes = (diff >= threshold).astype(np.float64)
    off_spikes = (diff <= -threshold).astype(np.float64)
    return on_spikes, off_spikes
