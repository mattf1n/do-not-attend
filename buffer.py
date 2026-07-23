import torch
from contextlib import nullcontext
from transformers.utils import generic as hf_generic
from transformers.models.olmo3 import modeling_olmo3 as olmo3_mod
from nnterp import StandardizedTransformer

# nnterp validation uses model.scan() on meta tensors. OLMo-3 RoPE calls
# maybe_autocast(device_type=x.device.type), and PyTorch rejects "meta".
# Map unsupported device types to a no-op so scan succeeds.
_AUTOCAST_DEVICES = {"cpu", "cuda", "mps", "xpu", "hpu"}
_orig_maybe_autocast = hf_generic.maybe_autocast


def _maybe_autocast_meta_safe(device_type="cuda", dtype=None, enabled=True, cache_enabled=None):
    if device_type not in _AUTOCAST_DEVICES:
        return nullcontext()
    return _orig_maybe_autocast(
        device_type=device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
    )


hf_generic.maybe_autocast = _maybe_autocast_meta_safe
olmo3_mod.maybe_autocast = _maybe_autocast_meta_safe  # already imported by name

MODEL = "allenai/Olmo-3-1025-7B"

# enable_attention_probs=True forces eager + runs nnterp's load-time validation
model = StandardizedTransformer(MODEL, enable_attention_probs=True)
print("loaded:", model.num_layers, "layers,", model.num_heads, "heads")

# where does the attention-prob hook actually land in OLMo's forward?
model.attention_probabilities.print_source()

with model.trace("The quick brown fox jumps over the lazy dog"):
    attn = model.attention_probabilities[0].save()

print("shape:", tuple(attn.shape), "dtype:", attn.dtype)
rowsums = attn.float().sum(-1)
max_dev = (rowsums - 1).abs().max().item()
# atol scaled to the stored dtype: bf16/fp16 softmax rounding drifts well past 1e-3.
tol = 1e-6 if attn.dtype == torch.float32 else 5e-2
print(f"max |rowsum - 1|: {max_dev:.2e}  (tol {tol:g})")
print("rows sum to 1:", max_dev < tol)
print("min/max:", attn.float().min().item(), attn.float().max().item())


'''
[OLD       (get_attentions + aggregate_old)]
  wall time:         546.487 s
  RSS peak Δ:       9.92 GB
  Python heap peak: 11.63 MB

[NEW       (get_attentions + aggregate_new)]
  wall time:         541.643 s
  RSS peak Δ:       7.51 GB
  Python heap peak: 11.63 MB

[STREAMING (get_attentions_streaming)]
  wall time:        2544.377 s
  RSS peak Δ:       7.06 GB
  Python heap peak: 11.60 MB


'''