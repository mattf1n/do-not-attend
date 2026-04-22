"""
Compare the original (full-tuple) and streaming aggregation pipelines for
multi-token word attentions.

Verifies:
  1. Output equality — the two pipelines must produce numerically identical
     dicts (within fp16 round-off).
  2. Peak memory — measured with tracemalloc (Python heap) and resident
     RSS deltas (process-wide, captures torch tensor allocations too).
  3. Wall-clock time.

Usage:
    python compare_streaming.py                          # uses the_multi.json + Enron sample
    python compare_streaming.py --tokens 4000            # ignore JSON, derive map from a fresh sample
    python compare_streaming.py --text-file mytext.txt   # use the_multi.json with custom text
"""

import argparse
import gc
import json
import os
import time
import tracemalloc

import torch

from analysis import (
    MultiTokenWordAggregator,
    aggregate_multi_token_word_attentions,
    get_attentions_streaming,
)
from model import get_attentions, get_model
from tokenization import get_multi_token_words

DEFAULT_MULTI_WORD_MAP = "the_multi.json"


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rss_bytes() -> int:
    """Resident set size of the current process, in bytes (Linux /proc)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class MemoryProbe:
    """Sample RSS at high frequency in a background thread to catch peaks."""

    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak = 0
        self.start = 0
        self._stop = False
        self._thread = None

    def __enter__(self):
        import threading
        gc.collect()
        self.start = _rss_bytes()
        self.peak = self.start
        self._stop = False

        def _run():
            while not self._stop:
                rss = _rss_bytes()
                if rss > self.peak:
                    self.peak = rss
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        if self._thread is not None:
            self._thread.join()
        rss = _rss_bytes()
        if rss > self.peak:
            self.peak = rss

    @property
    def delta_peak(self) -> int:
        return self.peak - self.start


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


# ─────────────────────────────────────────────────────────────────────────────
# Equality check
# ─────────────────────────────────────────────────────────────────────────────


def assert_results_equal(a: dict, b: dict, atol: float = 1e-3, rtol: float = 1e-3):
    """
    Raise AssertionError if the two output dicts differ in structure or values.

    Tolerances are loose because both pipelines cast to fp16 at slightly
    different moments, so bit-exactness is not guaranteed — but values
    should be within fp16 round-off.
    """
    assert set(a.keys()) == set(b.keys()), (
        f"Word sets differ. Only in A: {set(a) - set(b)}; only in B: {set(b) - set(a)}"
    )
    max_abs_diff = 0.0
    for word in a:
        occ_a = a[word]["occurrences"]
        occ_b = b[word]["occurrences"]
        assert len(occ_a) == len(occ_b), f"Occurrence count mismatch for '{word}'"
        for i, (oa, ob) in enumerate(zip(occ_a, occ_b)):
            assert oa["token_indices"] == ob["token_indices"], (
                f"token_indices mismatch for '{word}' occ {i}"
            )
            layers_a = oa["attentions"]["layers"]
            layers_b = ob["attentions"]["layers"]
            assert len(layers_a) == len(layers_b), (
                f"Layer count mismatch for '{word}' occ {i}: "
                f"{len(layers_a)} vs {len(layers_b)}"
            )
            for li, (la, lb) in enumerate(zip(layers_a, layers_b)):
                heads_a = la["heads"]
                heads_b = lb["heads"]
                assert len(heads_a) == len(heads_b), (
                    f"Head count mismatch for '{word}' occ {i} layer {li}"
                )
                for hi, (ha, hb) in enumerate(zip(heads_a, heads_b)):
                    ta = ha if isinstance(ha, torch.Tensor) else torch.as_tensor(ha)
                    tb = hb if isinstance(hb, torch.Tensor) else torch.as_tensor(hb)
                    ta = ta.float()
                    tb = tb.float()
                    diff = (ta - tb).abs().max().item()
                    if diff > max_abs_diff:
                        max_abs_diff = diff
                    if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
                        raise AssertionError(
                            f"Value mismatch for '{word}' occ {i} layer {li} head {hi}: "
                            f"max abs diff {diff:.2e} (atol={atol}, rtol={rtol})\n"
                            f"  A: {ta.tolist()}\n  B: {tb.tolist()}"
                        )
    print(f"  outputs match (max abs diff across all heads: {max_abs_diff:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# Pipelines under test
# ─────────────────────────────────────────────────────────────────────────────


def run_original(text, model, tokenizer, multi_word_map):
    """Existing two-step pipeline: full attentions tuple, then aggregate."""
    attentions = get_attentions(text, model, tokenizer)
    result = aggregate_multi_token_word_attentions(attentions, multi_word_map)
    del attentions
    return result


def run_streaming(text, model, tokenizer, multi_word_map):
    """New streaming pipeline (hook + MultiTokenWordAggregator)."""
    return get_attentions_streaming(text, model, tokenizer, multi_word_map)


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


def benchmark(name, fn, *args, **kwargs):
    """Run fn under tracemalloc + RSS probe + wall-clock timer."""
    gc.collect()
    tracemalloc.start()
    probe = MemoryProbe()
    t0 = time.perf_counter()
    with probe:
        result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\n[{name}]")
    print(f"  wall time:        {elapsed:8.2f} s")
    print(f"  RSS peak Δ:       {_fmt_bytes(probe.delta_peak)}")
    print(f"  Python heap peak: {_fmt_bytes(py_peak)}")
    return result, elapsed, probe.delta_peak, py_peak


def _max_token_index(multi_word_map):
    """Largest token index referenced by any occurrence in the map."""
    return max(
        idx
        for info in multi_word_map.values()
        for occ in info["occurrences"]
        for idx in occ["token_indices"]
    )


def get_text(args, tokenizer, multi_word_map=None):
    """
    Return text to run the model on.

    Priority:
      1. --text-file if given.
      2. --tokens (pull from the dataset).
      3. If a multi_word_map is supplied, pull enough tokens from the dataset
         to cover its largest token index.
      4. Built-in synthetic text (only used when no map is given either).
    """
    if args.text_file is not None:
        with open(args.text_file, encoding="utf-8") as f:
            return f.read()

    if args.tokens is not None:
        from data import get_data_samples
        print(f"Loading ~{args.tokens} tokens from component={args.component!r}...")
        return get_data_samples(
            component=args.component,
            max_tokens=args.tokens,
            type="string",
        )

    if multi_word_map is not None:
        from data import get_data_samples
        # Need enough tokens to cover the highest index referenced by the map,
        # plus a small buffer for tokenizer differences across samples.
        needed = _max_token_index(multi_word_map) + 64
        print(
            f"Loading ~{needed} tokens from component={args.component!r} "
            f"to cover map's max token index..."
        )
        return get_data_samples(
            component=args.component,
            max_tokens=needed,
            type="string",
        )

    return (
        "The pneumonoultramicroscopicsilicovolcanoconiosis researcher "
        "studied antidisestablishmentarianism in supercalifragilisticexpialidocious "
        "detail. The pneumonoultramicroscopicsilicovolcanoconiosis researcher "
        "again confirmed antidisestablishmentarianism."
    )


def _load_multi_word_map(path):
    """
    Load a multi_word_map JSON. Accepts either:
      - a flat {word: {occurrences: [...]}} dump (e.g. the_multi.json), or
      - an output dict with {"main_data": {...}, "text": "..."} (returns text too).

    Returns (multi_word_map, text_or_None).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "main_data" in data:
        return data["main_data"], data.get("text")
    return data, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi-word-path", type=str, default=DEFAULT_MULTI_WORD_MAP,
        help=(
            "Path to a multi_word_map JSON "
            f"(default: {DEFAULT_MULTI_WORD_MAP}). "
            "Use --derive-map to ignore this and recompute from the text instead."
        ),
    )
    parser.add_argument(
        "--derive-map", action="store_true",
        help="Ignore --multi-word-path and derive the map from the text via get_multi_token_words.",
    )
    parser.add_argument(
        "--text-file", type=str, default=None,
        help="Path to a text file to run the model on. If omitted, samples from the dataset.",
    )
    parser.add_argument(
        "--tokens", type=int, default=None,
        help="Number of tokens to pull from the dataset when --text-file is not given.",
    )
    parser.add_argument(
        "--component", type=str, default="Enron Emails",
        help="Pile component to sample from (default matches the_multi.json's source).",
    )
    parser.add_argument(
        "--skip-original", action="store_true",
        help="Skip the original pipeline (useful if it OOMs).",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-3,
        help="Absolute tolerance for output comparison.",
    )
    args = parser.parse_args()

    print("Loading model + tokenizer...")
    model, tokenizer = get_model()
    model.eval()

    preloaded_map = None
    embedded_text = None
    if not args.derive_map:
        print(f"Loading multi_word_map from {args.multi_word_path}")
        preloaded_map, embedded_text = _load_multi_word_map(args.multi_word_path)

    if embedded_text is not None and args.text_file is None and args.tokens is None:
        text = embedded_text
        print("Using text embedded in the multi-word JSON.")
    else:
        text = get_text(args, tokenizer, multi_word_map=preloaded_map)
    n_tokens = len(tokenizer(text)["input_ids"])
    print(f"Input length: {n_tokens} tokens")

    if preloaded_map is not None:
        multi_word_map = preloaded_map
        max_idx = _max_token_index(multi_word_map)
        if max_idx >= n_tokens:
            raise ValueError(
                f"multi_word_map references token index {max_idx} but the loaded "
                f"text only tokenizes to {n_tokens} tokens. Provide a longer "
                f"--text-file or increase --tokens so the indices are valid."
            )
    else:
        multi_word_map = get_multi_token_words(text, tokenizer)
    n_words = len(multi_word_map)
    n_occs = sum(len(v["occurrences"]) for v in multi_word_map.values())
    print(f"Multi-token words: {n_words} unique, {n_occs} occurrences")

    if not multi_word_map:
        print("No multi-token words found — nothing to compare. Exiting.")
        return

    original_result = None
    if not args.skip_original:
        original_result, t_orig, rss_orig, py_orig = benchmark(
            "ORIGINAL  (get_attentions + aggregate_multi_token_word_attentions)",
            run_original, text, model, tokenizer, multi_word_map,
        )
        gc.collect()
    else:
        t_orig = rss_orig = py_orig = None

    streaming_result, t_stream, rss_stream, py_stream = benchmark(
        "STREAMING (get_attentions_streaming)",
        run_streaming, text, model, tokenizer, multi_word_map,
    )

    print("\n--- equality ---")
    if original_result is None:
        print("  (skipped — original pipeline was not run)")
    else:
        assert_results_equal(original_result, streaming_result, atol=args.atol)

    if t_orig is not None:
        print("\n--- summary ---")
        speedup = t_orig / t_stream if t_stream > 0 else float("inf")
        rss_ratio = (rss_orig / rss_stream) if rss_stream > 0 else float("inf")
        print(f"  time:       original {t_orig:.2f}s vs streaming {t_stream:.2f}s "
              f"({speedup:.2f}x)")
        print(f"  RSS peak Δ: original {_fmt_bytes(rss_orig)} vs streaming "
              f"{_fmt_bytes(rss_stream)} ({rss_ratio:.2f}x)")
        print(f"  py heap:    original {_fmt_bytes(py_orig)} vs streaming "
              f"{_fmt_bytes(py_stream)}")


if __name__ == "__main__":
    main()
