"""SpikeMamba Deep Architecture: Hybrid Neuromorphic SNN-SSM Layers."""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np

from .ssm import SelectiveSSM
from .neurons import LIFNeuron, AdaptiveLIFNeuron, LIFConfig


class SpikeMambaBlock:
    """Core SpikeMamba Residual Block.
    
    Dataflow:
        Spike In -> 1D Depthwise Conv -> LIF Gating -> Selective SSM -> Residual Add -> Spiking Output
    """

    def __init__(self, d_model: int = 64, d_state: int = 16, d_conv: int = 4, seed: int = 42):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        rng = np.random.default_rng(seed)
        
        # 1D Causal Convolution kernel (d_model, d_conv)
        self.conv_weight = rng.normal(0.0, 0.1, size=(d_model, d_conv))
        self.conv_bias = np.zeros((d_model,), dtype=np.float64)
        
        # Selective SSM layer
        self.ssm = SelectiveSSM(d_model=d_model, d_state=d_state, seed=seed + 1)
        
        # LIF Spiking activation units
        self.lif_gate = LIFNeuron(size=d_model, config=LIFConfig(tau_m=15.0, v_th=1.0))
        self.lif_out = LIFNeuron(size=d_model, config=LIFConfig(tau_m=20.0, v_th=0.8))
        
        # Projection matrices
        self.in_proj = rng.normal(0.0, 0.05, size=(d_model, 2 * d_model))
        self.out_proj = rng.normal(0.0, 0.05, size=(d_model, d_model))

    def causal_conv1d(self, x: np.ndarray) -> np.ndarray:
        """Causal depthwise 1D convolution over sequence dimension.
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            conv_out: (batch, seq_len, d_model)
        """
        batch, seq_len, dim = x.shape
        # Zero pad left by (d_conv - 1)
        pad_left = self.d_conv - 1
        x_padded = np.pad(x, ((0, 0), (pad_left, 0), (0, 0)), mode="constant")
        
        out = np.zeros_like(x)
        for t in range(seq_len):
            # slice of shape (batch, d_conv, dim)
            window = x_padded[:, t : t + self.d_conv, :]
            # depthwise elementwise multiplication with flipped kernel
            w_flipped = self.conv_weight[:, ::-1]  # (dim, d_conv)
            # window: (batch, d_conv, dim) -> einsum with (dim, d_conv)
            res = np.einsum("bkd,dk->bd", window, w_flipped) + self.conv_bias[None, :]
            out[:, t, :] = res
        return out

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Forward pass through SpikeMamba Block.
        
        Args:
            x: Spiking or continuous input sequence (batch, seq_len, d_model)
        Returns:
            spikes_out: Output binary spike events (batch, seq_len, d_model)
            analog_out: Continuous latent projection (batch, seq_len, d_model)
            metrics: Dictionary of neuromorphic energy metrics (spike sparsity, dynamic power)
        """
        # In-projection to 2 * d_model
        projected = np.einsum("bld,de->ble", x, self.in_proj)
        u, z = np.split(projected, 2, axis=-1)  # split branch and gate
        
        # Causal convolution on branch
        u_conv = self.causal_conv1d(u)
        
        # Neuromorphic LIF Gating
        gate_spikes, _ = self.lif_gate.forward_sequence(u_conv)
        
        # Selective SSM on spike-modulated branch
        ssm_in = gate_spikes * z
        ssm_out, _ = self.ssm.forward(ssm_in)
        
        # Projection back to d_model
        projected_out = np.einsum("bld,de->ble", ssm_out, self.out_proj)
        
        # Residual connection
        residual_sum = projected_out + x
        
        # Output spiking neuron firing
        spikes_out, v_out = self.lif_out.forward_sequence(residual_sum)
        
        sparsity = 1.0 - np.mean(spikes_out)
        # Neuromorphic energy proxy: synaptic operations (SynOps) proportional to non-zero events
        synops = int(np.sum(spikes_out) * self.d_model)
        
        metrics = {
            "spike_sparsity": float(sparsity),
            "firing_rate": float(np.mean(spikes_out)),
            "synops": synops,
            "estimated_energy_pj": float(synops * 0.9),  # ~0.9 pJ per neuromorphic MAC
        }
        return spikes_out, residual_sum, metrics


class SpikeMambaModel:
    """Multi-Layer Deep SpikeMamba Architecture for Neuromorphic Time-Series Reasoning."""

    def __init__(self, d_input: int, d_model: int = 64, d_state: int = 16, num_layers: int = 3, seed: int = 42):
        self.d_input = d_input
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        rng = np.random.default_rng(seed)
        self.in_embedding = rng.normal(0.0, 0.05, size=(d_input, d_model))
        
        self.layers = [
            SpikeMambaBlock(d_model=d_model, d_state=d_state, seed=seed + 10 * i)
            for i in range(num_layers)
        ]
        self.head = rng.normal(0.0, 0.05, size=(d_model, d_input))

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Full model forward pass.
        
        Args:
            x: Input sequence (batch, seq_len, d_input)
        Returns:
            y_pred: Output reconstructed/predicted signal (batch, seq_len, d_input)
            layer_metrics: List of metrics per layer
        """
        # Project to d_model
        h = np.einsum("bld,de->ble", x, self.in_embedding)
        
        layer_metrics = []
        for layer in self.layers:
            h_spikes, h_analog, metrics = layer.forward(h)
            h = h_spikes  # propagate discrete spikes across layers for event-driven efficiency
            layer_metrics.append(metrics)
            
        y_pred = np.einsum("bld,de->ble", h, self.head)
        return y_pred, layer_metrics
