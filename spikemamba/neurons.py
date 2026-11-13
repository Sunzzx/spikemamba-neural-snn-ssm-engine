"""Neuromorphic Spiking Neuron Dynamics with Surrogate Gradient Primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Callable
import numpy as np


class SurrogateGradient:
    """Base class for surrogate gradient functions in non-differentiable threshold operations."""
    
    @staticmethod
    def forward(v: np.ndarray, v_th: float) -> np.ndarray:
        return (v >= v_th).astype(np.float64)

    @staticmethod
    def backward(grad_output: np.ndarray, v: np.ndarray, v_th: float, alpha: float = 2.0) -> np.ndarray:
        raise NotImplementedError


class FastSigmoidSurrogate(SurrogateGradient):
    """Fast Sigmoid surrogate derivative: sigma'(x) = 1 / (1 + alpha * |x|)^2."""
    
    @staticmethod
    def backward(grad_output: np.ndarray, v: np.ndarray, v_th: float, alpha: float = 2.0) -> np.ndarray:
        x = v - v_th
        surrogate_grad = 1.0 / ((1.0 + alpha * np.abs(x)) ** 2)
        return grad_output * surrogate_grad


class ATanSurrogate(SurrogateGradient):
    """Arctangent surrogate derivative: sigma'(x) = 1 / (pi * (1 + (alpha * x)^2))."""
    
    @staticmethod
    def backward(grad_output: np.ndarray, v: np.ndarray, v_th: float, alpha: float = 2.0) -> np.ndarray:
        x = v - v_th
        surrogate_grad = 1.0 / (np.pi * (1.0 + (alpha * x) ** 2))
        return grad_output * surrogate_grad


@dataclass
class LIFConfig:
    tau_m: float = 20.0        # Membrane time constant (ms)
    v_th: float = 1.0          # Firing threshold (V)
    v_reset: float = 0.0       # Reset voltage (V)
    dt: float = 1.0            # Integration time step (ms)
    decay: float = 0.95        # Exponential decay factor: exp(-dt / tau_m)
    refractory_period: int = 0 # Timesteps of refractory lockout


class LIFNeuron:
    """Leaky Integrate-and-Fire (LIF) Neuron Layer with hard or soft reset."""

    def __init__(self, size: int, config: LIFConfig | None = None, reset_mode: str = "hard"):
        self.size = size
        self.config = config or LIFConfig()
        self.reset_mode = reset_mode
        self.v = np.full((size,), self.config.v_reset, dtype=np.float64)
        self.refractory_counters = np.zeros((size,), dtype=np.int32)

    def reset(self, batch_size: int | None = None) -> None:
        if batch_size is None:
            self.v = np.full((self.size,), self.config.v_reset, dtype=np.float64)
            self.refractory_counters = np.zeros((self.size,), dtype=np.int32)
        else:
            self.v = np.full((batch_size, self.size), self.config.v_reset, dtype=np.float64)
            self.refractory_counters = np.zeros((batch_size, self.size), dtype=np.int32)

    def step(self, i_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single timestep simulation.
        
        Args:
            i_in: Input current (shape: (size,) or (batch, size))
        Returns:
            spikes: Binary spike vector (0.0 or 1.0)
            v: Membrane potential after decay and input, before reset
        """
        active_mask = (self.refractory_counters == 0).astype(np.float64)
        
        # Membrane potential integration with exponential decay
        v_next = self.config.decay * self.v + i_in * active_mask
        
        # Fire
        spikes = (v_next >= self.config.v_th).astype(np.float64)
        
        # Reset mechanism
        if self.reset_mode == "hard":
            self.v = np.where(spikes > 0.0, self.config.v_reset, v_next)
        else:  # soft reset
            self.v = np.where(spikes > 0.0, v_next - self.config.v_th, v_next)
            
        if self.config.refractory_period > 0:
            self.refractory_counters = np.maximum(0, self.refractory_counters - 1)
            self.refractory_counters = np.where(spikes > 0.0, self.config.refractory_period, self.refractory_counters)
            
        return spikes, v_next

    def forward_sequence(self, i_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate full temporal sequence: (batch, seq_len, size)."""
        batch_size, seq_len, _ = i_seq.shape
        self.reset(batch_size=batch_size)
        
        spikes_list = []
        v_list = []
        for t in range(seq_len):
            s_t, v_t = self.step(i_seq[:, t, :])
            spikes_list.append(s_t)
            v_list.append(v_t)
            
        return np.stack(spikes_list, axis=1), np.stack(v_list, axis=1)


class AdaptiveLIFNeuron(LIFNeuron):
    """Adaptive Threshold Leaky Integrate-and-Fire Neuron.
    
    Dynamic threshold homeostasis: v_th[t] = v_th_0 + beta * w[t],
    where w[t] tracks spike frequency with decay tau_w.
    """

    def __init__(self, size: int, config: LIFConfig | None = None, beta: float = 0.5, tau_w_decay: float = 0.9):
        super().__init__(size, config)
        self.beta = beta
        self.tau_w_decay = tau_w_decay
        self.w = np.zeros((size,), dtype=np.float64)

    def reset(self, batch_size: int | None = None) -> None:
        super().reset(batch_size)
        if batch_size is None:
            self.w = np.zeros((self.size,), dtype=np.float64)
        else:
            self.w = np.zeros((batch_size, self.size), dtype=np.float64)

    def step(self, i_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        active_mask = (self.refractory_counters == 0).astype(np.float64)
        v_next = self.config.decay * self.v + i_in * active_mask
        dynamic_th = self.config.v_th + self.beta * self.w
        
        spikes = (v_next >= dynamic_th).astype(np.float64)
        
        if self.reset_mode == "hard":
            self.v = np.where(spikes > 0.0, self.config.v_reset, v_next)
        else:
            self.v = np.where(spikes > 0.0, v_next - dynamic_th, v_next)
            
        # Adapt threshold based on spike event
        self.w = self.tau_w_decay * self.w + spikes
        
        return spikes, v_next
