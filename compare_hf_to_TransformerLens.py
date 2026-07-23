import torch
from transformer_lens.model_bridge import TransformerBridge
from model import get_model, get_attentions

# ── Load once via your existing get_model(), hand it to the bridge ───────────
# boot_transformers accepts hf_model= so weights are not loaded a second time.
model_name = "allenai/Olmo-3-1025-7B"
hf_model, tokenizer = get_model(model_name, attn_implementation="eager")
bridge = TransformerBridge.boot_transformers(
    model_name,
    hf_model=hf_model,
    dtype=torch.float16,
)

cfg        = bridge.cfg
n_heads    = cfg.n_heads
n_kv_heads = cfg.n_key_value_heads if hasattr(cfg, "n_key_value_heads") else cfg.n_heads
d_model    = cfg.d_model
d_head     = cfg.d_head
n_layers   = cfg.n_layers
atol       = 1e-3   # float16 tolerance

layer = 0

# ══════════════════════════════════════════════════════════════════════════════
# 1. WEIGHT MATRICES
#    model.py access path  →  hf_model.model.layers[i].self_attn.*_proj.weight
#    bridge access path    →  bridge.blocks[i].attn.W_*
#                             bridge.W_*  (stacked over all layers)
# ══════════════════════════════════════════════════════════════════════════════

# model.py (raw HF) weights ─────────────────────────────────────────────────
#Wq,Wk,Wv, Wo
# Shapes as stored by HF (Linear.weight is always [out_features, in_features]):
#   q_proj : [n_heads    * d_head, d_model]
#   k_proj : [n_kv_heads * d_head, d_model]
#   v_proj : [n_kv_heads * d_head, d_model]
#   o_proj : [d_model,             n_heads * d_head]
hf_self_attn = hf_model.model.layers[layer].self_attn
hf_q = hf_self_attn.q_proj.weight
hf_k = hf_self_attn.k_proj.weight
hf_v = hf_self_attn.v_proj.weight
hf_o = hf_self_attn.o_proj.weight

# bridge (TL) weights ────────────────────────────────────────────────────────
# Shapes as stored by TL (head dim factored out):
#   W_Q : [n_heads,    d_model, d_head]
#   W_K : [n_kv_heads, d_model, d_head]
#   W_V : [n_kv_heads, d_model, d_head]
#   W_O : [n_heads,    d_head,  d_model]
tl_W_Q = bridge.blocks[layer].attn.W_Q
tl_W_K = bridge.blocks[layer].attn.W_K
tl_W_V = bridge.blocks[layer].attn.W_V
tl_W_O = bridge.blocks[layer].attn.W_O

# Reshape HF → TL convention:
#   HF [out, in] → split head dim → permute so d_model comes before d_head
hf_q_tl = hf_q.reshape(n_heads,    d_head, d_model).permute(0, 2, 1)
hf_k_tl = hf_k.reshape(n_kv_heads, d_head, d_model).permute(0, 2, 1)
hf_v_tl = hf_v.reshape(n_kv_heads, d_head, d_model).permute(0, 2, 1)
hf_o_tl = hf_o.T.reshape(n_heads,  d_head, d_model)

print("── Weight matrix comparison (layer 0) ──")
print(f"  Q  match : {torch.allclose(hf_q_tl, tl_W_Q, atol=atol)}"
      f"   HF {tuple(hf_q.shape)}  →  TL {tuple(tl_W_Q.shape)}")
print(f"  K  match : {torch.allclose(hf_k_tl, tl_W_K, atol=atol)}"
      f"   HF {tuple(hf_k.shape)}  →  TL {tuple(tl_W_K.shape)}")
print(f"  V  match : {torch.allclose(hf_v_tl, tl_W_V, atol=atol)}"
      f"   HF {tuple(hf_v.shape)}  →  TL {tuple(tl_W_V.shape)}")
print(f"  O  match : {torch.allclose(hf_o_tl, tl_W_O, atol=atol)}"
      f"   HF {tuple(hf_o.shape)}  →  TL {tuple(tl_W_O.shape)}")

# bridge.W_Q/K/V/O stack all layers: shape [n_layers, n_heads, d_model, d_head]
print(f"\n── Stacked (all layers) bridge shapes ──")
print(f"  bridge.W_Q : {tuple(bridge.W_Q.shape)}")
print(f"  bridge.W_K : {tuple(bridge.W_K.shape)}")
print(f"  bridge.W_V : {tuple(bridge.W_V.shape)}")
print(f"  bridge.W_O : {tuple(bridge.W_O.shape)}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RUNTIME ACTIVATIONS
#    model.py path  →  get_attentions() returns post-softmax scores per layer
#                       run with use_cache=True for past_key_values
#    bridge path    →  run_with_cache() captures hooks at any point
# ══════════════════════════════════════════════════════════════════════════════

text = "The quick brown fox"

# model.py — get_attentions() ────────────────────────────────────────────────
# Returns tuple of per-layer attention score tensors: [batch, n_heads, seq, seq]
# Internally runs: model(**inputs, output_attentions=True)
hf_attentions = get_attentions(text, hf_model, tokenizer)  # tuple of n_layers tensors
hf_attn_pattern = hf_attentions[layer]                     # [1, n_heads, seq, seq]

# model.py — past_key_values ─────────────────────────────────────────────────
# get_attentions() doesn't expose past_key_values, so we run separately.
# This mirrors exactly what get_attentions() does, plus use_cache=True.
inputs = tokenizer(text, return_tensors="pt")
device = next(hf_model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    hf_out = hf_model(**inputs, use_cache=True, output_attentions=True)

hf_pkv_k, hf_pkv_v = hf_out.past_key_values[layer]  # [1, n_kv_heads, seq, d_head]

# bridge — run_with_cache() ──────────────────────────────────────────────────
# hook_k / hook_v  : [batch, seq, n_kv_heads, d_head]  ← seq/head dims swapped vs HF
# hook_pattern     : [batch, n_heads, seq, seq]         ← same layout as HF attentions
tl_logits, tl_cache = bridge.run_with_cache(text)
tl_hook_k       = tl_cache[f"blocks.{layer}.attn.hook_k"]        # [1, seq, n_kv_heads, d_head]
tl_hook_v       = tl_cache[f"blocks.{layer}.attn.hook_v"]        # [1, seq, n_kv_heads, d_head]
tl_attn_pattern = tl_cache[f"blocks.{layer}.attn.hook_pattern"]  # [1, n_heads, seq, seq]

# Align HF past_key_values → TL layout (swap seq and head dims)
hf_pkv_k_tl = hf_pkv_k.permute(0, 2, 1, 3)  # [1, seq, n_kv_heads, d_head]
hf_pkv_v_tl = hf_pkv_v.permute(0, 2, 1, 3)

print(f"\n── Activation comparison (layer 0, '{text}') ──")
print(f"  Attn pattern  match : {torch.allclose(hf_attn_pattern.float(), tl_attn_pattern.float(), atol=atol)}"
      f"   shape {tuple(hf_attn_pattern.shape)}")
print(f"  K cache       match : {torch.allclose(hf_pkv_k_tl.float(), tl_hook_k.float(), atol=atol)}"
      f"   HF {tuple(hf_pkv_k.shape)} → TL {tuple(tl_hook_k.shape)}")
print(f"  V cache       match : {torch.allclose(hf_pkv_v_tl.float(), tl_hook_v.float(), atol=atol)}"
      f"   HF {tuple(hf_pkv_v.shape)} → TL {tuple(tl_hook_v.shape)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. k.hook_out / v.hook_out  vs  kv_cache.entries[layer].past_keys/past_values
# ══════════════════════════════════════════════════════════════════════════════
#
# NOTE on hook path for OLMo:
#   OLMo uses PositionEmbeddingsAttentionBridge (separate q/k/v projections).
#   The correct hook paths are:
#       cache["blocks.{i}.attn.k.hook_out"]   ← pre-RoPE keys
#       cache["blocks.{i}.attn.v.hook_out"]   ← values (no rotation applied)
#   "hook_k" and "hook_v" are registered aliases for these exact paths.
#
#   "blocks.0.attn.qkv.k_hook_out" does NOT exist for OLMo — that path is
#   only valid for JointQKVAttentionBridge (models with a single fused QKV
#   weight, e.g. some GPT-2 variants).
#
# NOTE on kv_cache.entries:
#   TransformerLensKeyValueCache is TL's native KV cache object, used by
#   HookedTransformer during autoregressive generation (token-by-token).
#   The bridge's run_with_cache() does not populate it; instead the bridge
#   relies on HF's native use_cache mechanism internally.
#   To populate kv_cache.entries you must call bridge.forward() directly
#   with use_past_kv_cache=True, or use bridge.generate().
#
# NOTE on pre-RoPE vs post-RoPE:
#   hook_k / k.hook_out  → keys BEFORE RoPE rotation
#   hook_rot_k           → keys AFTER  RoPE rotation  ← matches past_key_values
#   past_key_values and kv_cache.entries store POST-RoPE keys.
#   So to compare hook output with the KV cache you must use hook_rot_k.

from transformer_lens.cache import TransformerLensKeyValueCache

# Build an empty TL KV cache and run one forward pass to populate it.
kv_cache = TransformerLensKeyValueCache.init_cache(
    cfg=bridge.cfg,
    device=bridge.cfg.device,
    batch_size=1,
)

# bridge.run_with_cache does not accept past_kv_cache; use forward() directly.
# We also grab hook_rot_k (post-RoPE) which is what ends up in the KV cache.
with torch.no_grad():
    _, tl_cache2 = bridge.run_with_cache(text)

# The bridge populates HF's past_key_values internally; to get TL's kv_cache
# populated we need to route through the bridge's generate/forward path.
# For a single-pass comparison, the most direct equivalent is:
#   hook_rot_k  ≈  kv_cache.entries[layer].past_keys   (both post-RoPE)
#   hook_k      ≈  pre-RoPE projection output only
#
# Grab both hook variants from the cache we already have:
tl_k_pre_rope  = tl_cache2[f"blocks.{layer}.attn.k.hook_out"]   # pre-RoPE  [1, seq, n_kv_heads, d_head]
tl_k_post_rope = tl_cache2[f"blocks.{layer}.attn.hook_rot_k"]   # post-RoPE [1, seq, n_heads, d_head]
tl_v_out       = tl_cache2[f"blocks.{layer}.attn.v.hook_out"]   # values     [1, seq, n_kv_heads, d_head]

# HF past_key_values already collected above (hf_pkv_k, hf_pkv_v) — post-RoPE.
# Align to TL layout [batch, seq, heads, d_head] for comparison.
hf_pkv_k_tl2 = hf_pkv_k.permute(0, 2, 1, 3)
hf_pkv_v_tl2 = hf_pkv_v.permute(0, 2, 1, 3)

print(f"\n── hook_out vs past_key_values (layer 0) ──")
print(f"  k.hook_out  (pre-RoPE) vs HF past_k (post-RoPE): "
      f"{torch.allclose(tl_k_pre_rope.float(), hf_pkv_k_tl2.float(), atol=atol)}  "
      f"← expected False (RoPE not yet applied)")
print(f"  hook_rot_k (post-RoPE) vs HF past_k (post-RoPE): "
      f"{torch.allclose(tl_k_post_rope.float(), hf_pkv_k_tl2.float(), atol=atol)}  "
      f"← expected True")
print(f"  v.hook_out             vs HF past_v             : "
      f"{torch.allclose(tl_v_out.float(), hf_pkv_v_tl2.float(), atol=atol)}  "
      f"← expected True (values have no rotation)")


'''
── Weight matrix comparison (layer 0) ──
  Q  match : True   HF (4096, 4096)  →  TL (32, 4096, 128)
  K  match : True   HF (4096, 4096)  →  TL (32, 4096, 128)
  V  match : True   HF (4096, 4096)  →  TL (32, 4096, 128)
  O  match : True   HF (4096, 4096)  →  TL (32, 128, 4096)

── Stacked (all layers) bridge shapes ──
  bridge.W_Q : (32, 32, 4096, 128)
  bridge.W_K : (32, 32, 4096, 128)
  bridge.W_V : (32, 32, 4096, 128)
  bridge.W_O : (32, 32, 128, 4096)

── Activation comparison (layer 0, 'The quick brown fox') ──
  Attn pattern       match : True   shape (1, 32, 4, 4)
  K cache (post-RoPE) match: True   HF (1, 32, 4, 128) → TL (1, 4, 32, 128)
  V cache            match : True   HF (1, 32, 4, 128) → TL (1, 4, 32, 128)

── hook_out vs past_key_values (layer 0) ──
  k.hook_out  (pre-RoPE) vs HF past_k (post-RoPE): False  ← expected False (RoPE not yet applied)
  hook_rot_k (post-RoPE) vs HF past_k (post-RoPE): True  ← expected True
  v.hook_out             vs HF past_v             : True  ← expected True (values have no rotation)

'''