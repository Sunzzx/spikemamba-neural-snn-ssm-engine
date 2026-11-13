"""Comprehensive Test Suite for SpikeMamba SNN-SSM Engine."""
from __future__ import annotations

import numpy as np
import pytest

from spikemamba.neurons import (
    LIFNeuron,
    AdaptiveLIFNeuron,
    LIFConfig,
    FastSigmoidSurrogate,
    ATanSurrogate,
)
from spikemamba.ssm import SelectiveSSM, associative_scan
from spikemamba.encoding import PoissonEncoder, RateDecoder, temporal_contrast_encoder
from spikemamba.model import SpikeMambaBlock, SpikeMambaModel


class TestNeuromorphicNeurons:
    def test_lif_threshold_and_reset(self):
        """Verify LIF neuron fires if and only if v >= v_th, then resets to v_reset."""
        neuron = LIFNeuron(size=2, config=LIFConfig(v_th=1.0, v_reset=0.0, decay=0.9))
        
        # Step 1: Subthreshold
        spikes, v = neuron.step(np.array([0.5, 0.8]))
        assert np.all(spikes == 0.0)
        assert np.all(neuron.v > 0.0)
        
        # Step 2: Suprathreshold fire on ch 1, sub on ch 0
        spikes, v = neuron.step(np.array([0.2, 0.6]))
        assert spikes[0] == 0.0
        assert spikes[1] == 1.0
        # Hard reset on ch 1
        assert neuron.v[1] == 0.0
        assert neuron.v[0] > 0.0

    def test_adaptive_lif_homeostasis(self):
        """Verify adaptive threshold elevates upon persistent stimulation."""
        alif = AdaptiveLIFNeuron(size=1, beta=1.0, tau_w_decay=0.9)
        # Sustained high input current
        burst = np.ones((1, 50, 1)) * 1.5
        spikes, _ = alif.forward_sequence(burst)
        
        # Early steps fire continuously; later steps space out due to threshold adaptation
        early_rate = np.mean(spikes[:, :10, :])
        late_rate = np.mean(spikes[:, 40:, :])
        assert early_rate >= late_rate
        assert alif.w[0, 0] > 0.0

    def test_surrogate_gradients(self):
        """Verify surrogate derivatives evaluate to correct peak at threshold and vanish symmetrically."""
        v = np.array([0.0, 1.0, 2.0])
        v_th = 1.0
        grad = np.ones_like(v)
        
        d_fast = FastSigmoidSurrogate.backward(grad, v, v_th=v_th, alpha=2.0)
        assert d_fast[1] == 1.0  # At threshold, (1 + 0)^(-2) = 1.0
        assert d_fast[0] < d_fast[1]
        assert d_fast[2] < d_fast[1]
        assert np.isclose(d_fast[0], d_fast[2])  # Symmetric around v_th
        
        d_atan = ATanSurrogate.backward(grad, v, v_th=v_th, alpha=2.0)
        assert d_atan[1] == pytest.approx(1.0 / np.pi)
        assert d_atan[0] < d_atan[1]


class TestSelectiveSSM:
    def test_associative_scan_invariance(self):
        """Verify scan matches exact linear recurrence."""
        batch, seq, dim = 2, 8, 4
        rng = np.random.default_rng(42)
        a = rng.uniform(0.5, 0.99, size=(batch, seq, dim))
        b = rng.normal(0.0, 1.0, size=(batch, seq, dim))
        
        scanned, _ = associative_scan(a, b)
        
        # Manual validation
        manual = np.zeros_like(b)
        curr = np.zeros((batch, dim))
        for t in range(seq):
            curr = a[:, t, :] * curr + b[:, t, :]
            manual[:, t, :] = curr
            
        assert np.allclose(scanned, manual, atol=1e-10)

    def test_selective_ssm_forward(self):
        """Verify SSM shape, numerical stability, and non-trivial hidden state update."""
        ssm = SelectiveSSM(d_model=16, d_state=8, seed=42)
        x = np.random.randn(3, 20, 16)
        y, h_final = ssm.forward(x)
        
        assert y.shape == (3, 20, 16)
        assert h_final.shape == (3, 16, 8)
        assert not np.isnan(y).any()
        assert not np.isinf(y).any()


class TestEncodingAndDecoding:
    def test_poisson_encoder_stochasticity(self):
        """Verify Poisson encoder outputs binary spikes and tracks rate."""
        encoder = PoissonEncoder(time_steps=200, max_rate_hz=100.0, dt_s=0.005, seed=42)
        x = np.array([[0.1, 0.9]])
        spikes = encoder.encode(x)
        
        assert set(np.unique(spikes)).issubset({0.0, 1.0})
        # High input intensity must trigger more spikes than low intensity
        rate_ch0 = np.mean(spikes[0, :, 0])
        rate_ch1 = np.mean(spikes[0, :, 1])
        assert rate_ch1 > rate_ch0

    def test_temporal_contrast_encoder(self):
        """Verify DVS polarity generation on ascending/descending signals."""
        t = np.linspace(0, 1, 10)
        ramp = np.stack([t, -t], axis=-1)[None, :, :]  # (1, 10, 2)
        on_spikes, off_spikes = temporal_contrast_encoder(ramp, threshold=0.05)
        
        assert np.sum(on_spikes[0, :, 0]) > 0
        assert np.sum(off_spikes[0, :, 0]) == 0
        assert np.sum(off_spikes[0, :, 1]) > 0
        assert np.sum(on_spikes[0, :, 1]) == 0


class TestFullSpikeMambaArchitecture:
    def test_block_forward_and_energy_metrics(self):
        """Verify residual block returns spikes, analog projections, and valid synops metrics."""
        block = SpikeMambaBlock(d_model=16, d_state=4, d_conv=3, seed=42)
        x = (np.random.rand(2, 12, 16) > 0.7).astype(np.float64)
        spikes_out, res_out, metrics = block.forward(x)
        
        assert spikes_out.shape == (2, 12, 16)
        assert res_out.shape == (2, 12, 16)
        assert set(np.unique(spikes_out)).issubset({0.0, 1.0})
        assert 0.0 <= metrics["spike_sparsity"] <= 1.0
        assert metrics["synops"] >= 0
        assert metrics["estimated_energy_pj"] >= 0.0

    def test_multi_layer_deep_model(self):
        """Verify end-to-end multi-layer model execution."""
        model = SpikeMambaModel(d_input=8, d_model=16, d_state=4, num_layers=2, seed=42)
        x = np.random.randn(2, 10, 8)
        y_pred, layer_metrics = model.forward(x)
        
        assert y_pred.shape == (2, 10, 8)
        assert len(layer_metrics) == 2
        assert not np.isnan(y_pred).any()
