"""Model stubs: forecaster F0/F1, parameter head H, differentiable routing.

Modules to be filled in Milestone 2 (§12.2):

- ``forecaster.py`` — encoder–decoder LSTM (F0) and Transformer/TFT variant.
- ``physics.py`` — F1 = F0 + soft constraints (mass balance, non-negativity,
  monotonic rainfall → runoff, recession behaviour).
- ``parameter_head.py`` — H : ``a_b → θ_b`` (recession, storage, lag / roughness).
- ``routing.py`` — differentiable routing / δ-model conditioned on θ.
"""
