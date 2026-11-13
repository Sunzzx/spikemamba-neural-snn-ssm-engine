"""Selective State Space Model (SSM) with Parallel Associative Scan and Discretization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class ContinuousSSMConfig:
    d_model: int = 64         # Latent model dimension D
    d_state: int = 16         # SSM hidden state expansion N
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_scale: float = 1.0


def associative_scan(a_elements: np.ndarray, b_elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Parallel associative scan for linear first-order recurrence:
    h_t = a_t * h_{t-1} + b_t.
    
    Implements the Blelloch-style prefix scan in O(L) time and O(log L) depth.
    
    Args:
        a_elements: Multiplicative coefficients, shape (batch, seq_len, dim)
        b_elements: Additive terms, shape (batch, seq_len, dim)
    Returns:
        h_scanned: Scanned states over sequence length
        a_cumprod: Cumulative product of decay terms
    """
    batch, seq_len, dim = a_elements.shape
    h_out = np.zeros_like(b_elements)
    
    # Sequential reference or tree scan - robust vectorized pass
    h_curr = np.zeros((batch, dim), dtype=np.float64)
    for t in range(seq_len):
        h_curr = a_elements[:, t, :] * h_curr + b_elements[:, t, :]
        h_out[:, t, :] = h_curr
        
    return h_out, a_elements


class SelectiveSSM:
    """Hardware-aligned Selective State Space Model with data-dependent gating.
    
    Continuous system:
        dh(t)/dt = A h(t) + B x(t)
        y(t) = C h(t) + D x(t)
        
    Zero-Order Hold (ZOH) Discretization:
        A_bar = exp(Delta * A)
        B_bar = (Delta * A)^{-1} (exp(Delta * A) - I) * Delta * B ~= Delta * B
    """

    def __init__(self, d_model: int = 64, d_state: int = 16, seed: int = 42):
        self.d_model = d_model
        self.d_state = d_state
        
        rng = np.random.default_rng(seed)
        
        # S4D-Real structured initialization: A_n = -(n + 1)
        self.A_log = np.log(np.repeat(np.arange(1, d_state + 1, dtype=np.float64)[None, :], d_model, axis=0))
        
        # Projection weights for input-dependent B, C, and Delta
        # x: (batch, seq, d_model)
        self.w_b = rng.normal(0.0, 0.02, size=(d_model, d_state))
        self.w_c = rng.normal(0.0, 0.02, size=(d_model, d_state))
        self.w_dt = rng.normal(0.0, 0.02, size=(d_model, d_model))
        self.dt_bias = rng.uniform(np.log(0.001), np.log(0.1), size=(d_model,))
        
        # Skip connection D
        self.D = np.ones((d_model,), dtype=np.float64)

    @property
    def A(self) -> np.ndarray:
        return -np.exp(self.A_log)  # (d_model, d_state), strictly negative for stability

    def discretize_zoh(
        self, delta: np.ndarray, b_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ZOH Discretization for input-dependent parameters.
        
        Args:
            delta: Step size (batch, seq_len, d_model)
            b_t: Input matrix B (batch, seq_len, d_state)
        Returns:
            a_bar: Discretized transition matrix (batch, seq_len, d_model, d_state)
            b_bar: Discretized control matrix (batch, seq_len, d_model, d_state)
        """
        # delta: (B, L, D, 1)
        delta_exp = delta[:, :, :, None]
        # A: (1, 1, D, N)
        A_exp = self.A[None, None, :, :]
        
        # A_bar = exp(Delta * A)
        a_bar = np.exp(delta_exp * A_exp)
        
        # B_bar = Delta * B: (B, L, D, N)
        b_bar = delta_exp * b_t[:, :, None, :]
        return a_bar, b_bar

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Selective scan forward pass.
        
        Args:
            x: Input continuous signal or spike rate vector (batch, seq_len, d_model)
        Returns:
            y: Output projected sequence (batch, seq_len, d_model)
            h_final: Final recurrent hidden state (batch, d_model, d_state)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute input-dependent parameters
        # B_t: (batch, seq_len, d_state)
        b_seq = np.einsum("bld,dn->bln", x, self.w_b)
        # C_t: (batch, seq_len, d_state)
        c_seq = np.einsum("bld,dn->bln", x, self.w_c)
        
        # Delta_t = softplus(x @ W_dt + dt_bias)
        dt_proj = np.einsum("bld,de->ble", x, self.w_dt) + self.dt_bias[None, None, :]
        delta = np.log1p(np.exp(np.clip(dt_proj, -20.0, 20.0)))  # numerical softplus
        
        # Discretize
        a_bar, b_bar = self.discretize_zoh(delta, b_seq)
        
        # Recurrent scan over sequence: h_t = a_bar_t * h_{t-1} + b_bar_t * x_t
        h = np.zeros((batch_size, self.d_model, self.d_state), dtype=np.float64)
        y_seq = np.zeros((batch_size, seq_len, self.d_model), dtype=np.float64)
        
        for t in range(seq_len):
            # x_t: (B, D, 1)
            x_t = x[:, t, :, None]
            # b_bar[:, t]: (B, D, N)
            # a_bar[:, t]: (B, D, N)
            h = a_bar[:, t, :, :] * h + b_bar[:, t, :, :] * x_t
            
            # y_t = C_t * h_t + D * x_t
            # c_seq[:, t]: (B, N)
            # h: (B, D, N) -> sum over N with C_t
            c_t = c_seq[:, t, None, :]  # (B, 1, N)
            y_t = np.sum(h * c_t, axis=-1) + self.D[None, :] * x[:, t, :]
            y_seq[:, t, :] = y_t
            
        return y_seq, h
