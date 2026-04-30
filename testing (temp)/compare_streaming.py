"""
Compare all four aggregation pipelines for multi-token word attentions.

Benchmarks (full pipeline — model forward pass included in each):
  1. OLD      : get_attentions + aggregate_multi_token_word_attentions_old
  2. NEW      : get_attentions + aggregate_multi_token_word_attentions
  3. STREAMING: get_attentions_streaming  (hook-based, one layer at a time)
  4. HEAD     : get_attentions_head_streaming  (patched forward, one head at a time)

Reports wall-clock time, RSS peak delta, and Python heap peak for each,
verifies that outputs are numerically equal, then prints a summary.

Usage:
    python compare_streaming.py                     # uses the_multi.json + Enron dataset
    python compare_streaming.py --tokens 2000       # override token budget
    python compare_streaming.py --text-file foo.txt # supply your own text
    python compare_streaming.py --skip old,new      # skip OOM-prone pipelines at large S
"""

import argparse
import gc
import json
import os
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from analysis import (
    aggregate_multi_token_word_attentions,
    aggregate_multi_token_word_attentions_old,
    get_attentions_streaming,
    get_attentions_head_streaming,
)
from model import get_attentions, get_model

DEFAULT_MULTI_WORD_PATH = "the_multi.json"


# ─────────────────────────────────────────────────────────────────────────────
# Memory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _rss_bytes() -> int:
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


def assert_results_equal(label_a, a, label_b, b, atol=1e-3, rtol=1e-3):
    assert set(a.keys()) == set(b.keys()), (
        f"[{label_a} vs {label_b}] Word sets differ. "
        f"Only in {label_a}: {set(a) - set(b)}; only in {label_b}: {set(b) - set(a)}"
    )
    max_abs_diff = 0.0
    for word in a:
        occ_a = a[word]["occurrences"]
        occ_b = b[word]["occurrences"]
        assert len(occ_a) == len(occ_b), (
            f"[{label_a} vs {label_b}] Occurrence count mismatch for '{word}'"
        )
        for i, (oa, ob) in enumerate(zip(occ_a, occ_b)):
            assert oa["token_indices"] == ob["token_indices"], (
                f"[{label_a} vs {label_b}] token_indices mismatch for '{word}' occ {i}"
            )
            layers_a = oa["attentions"]["layers"]
            layers_b = ob["attentions"]["layers"]
            assert len(layers_a) == len(layers_b), (
                f"[{label_a} vs {label_b}] Layer count mismatch for '{word}' occ {i}"
            )
            for li, (la, lb) in enumerate(zip(layers_a, layers_b)):
                heads_a, heads_b = la["heads"], lb["heads"]
                assert len(heads_a) == len(heads_b), (
                    f"[{label_a} vs {label_b}] Head count mismatch for '{word}' occ {i} layer {li}"
                )
                for hi, (ha, hb) in enumerate(zip(heads_a, heads_b)):
                    ta = (ha if isinstance(ha, torch.Tensor) else torch.as_tensor(ha)).float()
                    tb = (hb if isinstance(hb, torch.Tensor) else torch.as_tensor(hb)).float()
                    diff = (ta - tb).abs().max().item()
                    if diff > max_abs_diff:
                        max_abs_diff = diff
                    if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
                        raise AssertionError(
                            f"[{label_a} vs {label_b}] Value mismatch for '{word}' "
                            f"occ {i} layer {li} head {hi}: max abs diff {diff:.2e}\n"
                            f"  {label_a}: {ta.tolist()}\n  {label_b}: {tb.tolist()}"
                        )
    print(f"  {label_a} vs {label_b}: MATCH  (max abs diff {max_abs_diff:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────────────────────


def run_old(text, model, tokenizer, multi_word_map):
    attentions = get_attentions(text, model, tokenizer)
    result = aggregate_multi_token_word_attentions_old(attentions, multi_word_map)
    del attentions
    return result


def run_new(text, model, tokenizer, multi_word_map):
    attentions = get_attentions(text, model, tokenizer)
    result = aggregate_multi_token_word_attentions(attentions, multi_word_map)
    del attentions
    return result


def run_streaming(text, model, tokenizer, multi_word_map):
    return get_attentions_streaming(text, model, tokenizer, multi_word_map)


def run_head(text, model, tokenizer, multi_word_map):
    return get_attentions_head_streaming(text, model, tokenizer, multi_word_map)


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────


def benchmark(name, fn, *args, **kwargs):
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
    print(f"  wall time:        {elapsed:8.3f} s")
    print(f"  RSS peak Δ:       {_fmt_bytes(probe.delta_peak)}")
    print(f"  Python heap peak: {_fmt_bytes(py_peak)}")
    return result, elapsed, probe.delta_peak, py_peak


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _max_token_index(multi_word_map):
    return max(
        idx
        for info in multi_word_map.values()
        for occ in info["occurrences"]
        for idx in occ["token_indices"]
    )


def load_multi_word_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "main_data" in data:
        return data["main_data"], data.get("text")
    return data, None


def get_text(args, tokenizer, multi_word_map):
    if args.text_file is not None:
        with open(args.text_file, encoding="utf-8") as f:
            return f.read()

    from data import get_data_samples
    needed = args.tokens if args.tokens is not None else _max_token_index(multi_word_map) + 64
    print(f"Loading ~{needed} tokens from component={args.component!r}...")
    return get_data_samples(component=args.component, max_tokens=needed, type="string")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-word-path", type=str, default=DEFAULT_MULTI_WORD_PATH)
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--component", type=str, default="Enron Emails")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument(
        "--skip", type=str, default="",
        help="Comma-separated list of pipelines to skip: old, new, streaming, head. "
             "Useful at large S where old/new OOM. Example: --skip old,new",
    )
    args = parser.parse_args()

    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    print(f"Loading multi_word_map from {args.multi_word_path}...")
    multi_word_map, embedded_text = load_multi_word_map(args.multi_word_path)
    n_words = len(multi_word_map)
    n_occs = sum(len(v["occurrences"]) for v in multi_word_map.values())
    print(f"Multi-token words: {n_words} unique, {n_occs} occurrences")

    print("Loading model + tokenizer...")
    model, tokenizer = get_model()
    model.eval()

    if embedded_text is not None and args.text_file is None and args.tokens is None:
        text = embedded_text
        print("Using text embedded in multi-word JSON.")
    else:
        text = get_text(args, tokenizer, multi_word_map)

    n_tokens = len(tokenizer(text)["input_ids"])
    print(f"Input length: {n_tokens} tokens")

    max_idx = _max_token_index(multi_word_map)
    if max_idx >= n_tokens:
        raise ValueError(
            f"multi_word_map references token index {max_idx} but text only has "
            f"{n_tokens} tokens. Increase --tokens or supply a longer --text-file."
        )

    # ── Benchmarks ────────────────────────────────────────────────────────────

    old_result = new_result = stream_result = head_result = None
    t_old = t_new = t_stream = t_head = None
    rss_old = rss_new = rss_stream = rss_head = None
    py_old = py_new = py_stream = py_head = None

    if "old" not in skip:
        old_result, t_old, rss_old, py_old = benchmark(
            "OLD       (get_attentions + aggregate_old)",
            run_old, text, model, tokenizer, multi_word_map,
        )
        gc.collect()
    else:
        print("[OLD] skipped")

    if "new" not in skip:
        new_result, t_new, rss_new, py_new = benchmark(
            "NEW       (get_attentions + aggregate_new)",
            run_new, text, model, tokenizer, multi_word_map,
        )
        gc.collect()
    else:
        print("[NEW] skipped")

    if "streaming" not in skip:
        stream_result, t_stream, rss_stream, py_stream = benchmark(
            "STREAMING (get_attentions_streaming)",
            run_streaming, text, model, tokenizer, multi_word_map,
        )
        gc.collect()
    else:
        print("[STREAMING] skipped")

    if "head" not in skip:
        head_result, t_head, rss_head, py_head = benchmark(
            "HEAD      (get_attentions_head_streaming)",
            run_head, text, model, tokenizer, multi_word_map,
        )
        gc.collect()
    else:
        print("[HEAD] skipped")

    # ── Equality ──────────────────────────────────────────────────────────────

    print("\n--- equality ---")
    results = {
        "OLD": old_result,
        "NEW": new_result,
        "STREAMING": stream_result,
        "HEAD": head_result,
    }
    available = [(k, v) for k, v in results.items() if v is not None]
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            label_a, res_a = available[i]
            label_b, res_b = available[j]
            assert_results_equal(label_a, res_a, label_b, res_b, atol=args.atol)

    if len(available) < 2:
        print("  (fewer than 2 pipelines ran — nothing to compare)")

    # ── Summary ───────────────────────────────────────────────────────────────

    print("\n--- summary ---")
    header = f"  {'':12s}  {'time (s)':>10}  {'RSS peak Δ':>12}  {'py heap peak':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    all_rows = [
        ("OLD",       t_old,    rss_old,    py_old),
        ("NEW",       t_new,    rss_new,    py_new),
        ("STREAMING", t_stream, rss_stream, py_stream),
        ("HEAD",      t_head,   rss_head,   py_head),
    ]
    for label, t, rss, py in all_rows:
        if t is None:
            print(f"  {label:12s}  {'(skipped)':>10}")
        else:
            print(f"  {label:12s}  {t:>10.3f}  {_fmt_bytes(rss):>12}  {_fmt_bytes(py):>14}")

    # Relative comparisons for pairs that ran
    ran = [(label, t, rss) for label, t, rss, _ in all_rows if t is not None]
    if len(ran) >= 2:
        ref_label, ref_t, ref_rss = ran[0]
        print()
        for label, t, rss in ran[1:]:
            t_ratio = ref_t / t if t else float("inf")
            rss_ratio = ref_rss / rss if rss else float("inf")
            print(f"  vs {ref_label} — {label}: {t_ratio:.2f}x time, {rss_ratio:.2f}x RSS")


if __name__ == "__main__":
    main()
