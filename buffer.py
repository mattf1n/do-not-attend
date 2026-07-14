"""
Layer-by-layer attention streaming: HuggingFace vs TransformerLens, benchmarked.

This file implements BOTH streaming approaches and measures, for each:
    - wall-clock time
    - peak CPU RAM (transient, above the model-weight baseline)
    - peak GPU RAM (if CUDA is available)

1) HF path  : analysis.get_attentions_streaming
              forward hook on each decoder layer's self_attn; reduces the
              [1,H,S,S] attention in place and nulls HF's tuple entry so the
              L attention tensors are never accumulated.

2) TL path  : get_attentions_streaming_tl (below)
              run_with_hooks on blocks.{i}.attn.hook_pattern; TL materializes
              one [1,H,S,S] pattern per layer, hands it to the hook, then frees
              it before the next layer. No accumulation (do NOT use run_with_cache).

Both feed the SAME analysis.MultiTokenWordAggregator, so their outputs are
directly comparable and the only difference measured is the library machinery.

NOTE: the head-by-head variant (analysis.get_attentions_head_streaming, peak
[S,S]) has no native TL equivalent — hook_pattern forces eager all-heads
attention — so it is intentionally not part of this comparison.

Run:  python buffer.py --tokens 1024
"""

import argparse
import gc
import threading
import time

import psutil
import torch

from analysis import MultiTokenWordAggregator, get_attentions_streaming


# ─────────────────────────────────────────────────────────────────────────────
# TransformerLens layer-by-layer streaming
# ─────────────────────────────────────────────────────────────────────────────
def get_attentions_streaming_tl(text, bridge, tokenizer, multi_word_map, stats=None):
    """
    TransformerLens equivalent of analysis.get_attentions_streaming.

    Streams one layer's attention pattern at a time via run_with_hooks and
    reduces it immediately, so peak attention memory stays at ~one [H,S,S]
    tensor regardless of n_layers.

    IMPORTANT: we tokenize with the SAME `tokenizer` used to build
    multi_word_map and feed the bridge a token tensor with prepend_bos=False.
    Passing a raw string instead lets TL prepend its own BOS, which shifts every
    token_index by one and makes scores disagree with the HF path.

    `stats` (optional dict) is filled with the observed pattern dtype/shape so
    the caller can see exactly what TL materializes per layer.
    """
    agg = MultiTokenWordAggregator(multi_word_map)

    def make_hook(layer_idx):
        def hook(pattern, hook):
            # pattern: [batch, n_heads, S, S] == HF attentions layout.
            if stats is not None and "pattern_dtype" not in stats:
                stats["pattern_dtype"] = str(pattern.dtype)
                stats["pattern_shape"] = tuple(pattern.shape)
            agg.add_layer(pattern)
            return None  # not returned -> freed before the next layer runs
        return hook

    fwd_hooks = [
        (f"blocks.{i}.attn.hook_pattern", make_hook(i))
        for i in range(bridge.cfg.n_layers)
    ]

    tokens = tokenizer(text, return_tensors="pt")["input_ids"].to(bridge.cfg.device)
    if stats is not None:
        stats["seq_len"] = int(tokens.shape[1])

    with torch.no_grad():
        bridge.run_with_hooks(
            tokens, return_type=None, prepend_bos=False, fwd_hooks=fwd_hooks
        )

    return agg.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────
class _PeakRSSSampler:
    """Background thread that records peak process RSS (bytes) while active."""

    def __init__(self, interval_s=0.02):
        self.interval_s = interval_s
        self._proc = psutil.Process()
        self._peak = 0
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self.interval_s)

    def __enter__(self):
        self._peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak_bytes(self):
        return self._peak


def _benchmark(label, fn, *args, **kwargs):
    """Run fn once, returning (result, elapsed_s, peak_cpu_mb, peak_gpu_mb)."""
    gc.collect()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    baseline_rss = psutil.Process().memory_info().rss
    t0 = time.perf_counter()
    with _PeakRSSSampler() as sampler:
        result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0

    peak_cpu_mb = max(0, sampler.peak_bytes - baseline_rss) / 1024**2
    peak_gpu_mb = (torch.cuda.max_memory_allocated() / 1024**2) if cuda else 0.0
    print(
        f"  {label:<28} time={elapsed:8.3f}s  "
        f"peak_cpu=+{peak_cpu_mb:9.1f} MB  peak_gpu={peak_gpu_mb:9.1f} MB"
    )
    return result, elapsed, peak_cpu_mb, peak_gpu_mb


def _build_text(tokenizer, target_tokens):
    """Repeat a passage until tokenized length >= target_tokens."""
    passage = (
        "The quick brown fox jumps over the lazy dog. "
        "Machine learning models process sequences of tokens efficiently. "
    )
    text = passage
    while len(tokenizer(text)["input_ids"]) < target_tokens:
        text += passage
    return text


def _build_dummy_word_map(tokenizer, text, n_occ=20):
    """Create n_occ single-token 'word' occurrences at valid, spread-out indices."""
    ids = tokenizer(text)["input_ids"]
    S = len(ids)
    # Valid attending rows require start = idx + 1 <= S, and we want at least
    # one attending row, so keep idx in [1, S - 2].
    step = max(1, (S - 3) // n_occ)
    occurrences = [{"token_indices": [idx]} for idx in range(1, S - 2, step)][:n_occ]
    return {"benchmark_word": {"occurrences": occurrences}}, S


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1024,
                        help="approximate sequence length to benchmark")
    parser.add_argument("--model", type=str, default="allenai/Olmo-3-1025-7B")
    args = parser.parse_args()

    from transformer_lens.model_bridge import TransformerBridge
    from model import get_model

    print(f"Loading {args.model} (eager) ...")
    hf_model, tokenizer = get_model(args.model, attn_implementation="eager")

    text = _build_text(tokenizer, args.tokens)
    multi_word_map, seq_len = _build_dummy_word_map(tokenizer, text)
    print(f"Sequence length: {seq_len} tokens, "
          f"{len(multi_word_map['benchmark_word']['occurrences'])} occurrences\n")

    # HF first: booting the bridge patches module forwards, so measure raw HF
    # before the bridge exists.
    print("── HuggingFace streaming ──")
    hf_out, *_ = _benchmark("get_attentions_streaming (HF)",
                            get_attentions_streaming,
                            text, hf_model, tokenizer, multi_word_map)

    print("\nBooting TransformerBridge (reuses HF weights) ...")
    bridge = TransformerBridge.boot_transformers(
        args.model, hf_model=hf_model, dtype=torch.float16
    )

    print("\n── TransformerLens streaming ──")
    tl_stats = {}
    tl_out, *_ = _benchmark("get_attentions_streaming_tl (TL)",
                            get_attentions_streaming_tl,
                            text, bridge, tokenizer, multi_word_map, stats=tl_stats)
    print(f"  TL pattern per layer: dtype={tl_stats.get('pattern_dtype')} "
          f"shape={tl_stats.get('pattern_shape')} seq_len={tl_stats.get('seq_len')}")

    # Sanity: outputs should be numerically close (float16 tolerance).
    hf_layers = hf_out["benchmark_word"]["occurrences"][0]["attentions"]["layers"]
    tl_layers = tl_out["benchmark_word"]["occurrences"][0]["attentions"]["layers"]
    max_diff = 0.0
    for hl, tl in zip(hf_layers, tl_layers):
        for hh, th in zip(hl["heads"], tl["heads"]):
            max_diff = max(max_diff, float((hh.float() - th.float()).abs().max()))
    print(f"\nMax |HF - TL| score diff: {max_diff:.4e} "
          f"(expect ~1e-3 at float16)")


if __name__ == "__main__":
    main()
