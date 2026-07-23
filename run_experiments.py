"""
run_experiments.py

Entry point for running all hypothesis experiments against a saved output JSON
(produced by main.py). Does not re-run the model except for exp3 (Q/K/V vectors).

Usage:
    python run_experiments.py                                             # uses default path
    python run_experiments.py output/my_output.json                       # single JSON
    python run_experiments.py output/my_output.json --exp 2 4             # run only exp 2 and 4
    python run_experiments.py --folder output/500_tokens/                 # all JSONs in folder
    python run_experiments.py --folder output/500_tokens/ --exp 4 5       # subset of exps
    python run_experiments.py output/my_output.json --exp 4 5 --word 'Colo'           # single word
    python run_experiments.py output/my_output.json --exp 4 5 --filter space_numbers  # word category
    python run_experiments.py --exp 8 --pt output/qkv_cache/16000_tokens/Pile-CC_16000tokens
    python run_experiments.py --exp 8 --pt output/qkv_cache/16000_tokens/

Experiments:
    2 — Box-whisker plots: attention score distributions per subtoken per head
    3 — Vector polar coordinates: geometric comparison of subtoken vectors (default k0 vs k1)
        (requires re-running via TransformerLens bridge to extract Q/K/V hooks)
    4 — Hypothesis rate analysis: heatmap + per-layer bar chart
    5 — Michelson contrast analysis: heatmap + per-layer bar chart
    8 — All QKV slot-pair polar heatmaps (C(6,2)=15 ordered pairs): theta and r
        for each word category, micro- and macro-averaged. Requires --pt (slot dir).
        Pair order: A=earlier subtoken, B=later; r = ||A|| / ||B||.

    Pooled experiments (--folder mode only, micro-averaged across all components):
    6 — Pooled hypothesis rate analysis (heatmap + per-layer bar)
    7 — Pooled Michelson contrast analysis (heatmap + per-layer bar)

Output folder structure:
    figures/<dataset_slug>/
        [all output PNGs saved flat, no subfolders]

    The dataset_slug is derived from the JSON filename, e.g.:
        output/10_paragraphs_2-token_max_word_output.json
        → figures/10_paragraphs_2-token_max_word/

    In --folder mode each JSON is nested under the input folder name, e.g.:
        output/500_tokens/PubMed_Abstracts_500tokens.json
        → figures/500_tokens/PubMed_Abstracts_500tokens/

    Exp 8 output uses the slot-dir name, e.g.:
        output/qkv_cache/16000_tokens/Pile-CC_16000tokens/
        → figures/Pile-CC_16000tokens/polar/<filter>/kv_theta_heatmap_micro.png
"""

import argparse
import json
import tempfile
from pathlib import Path
import os

from analysis import (
    compute_head_hypothesis_rates,
    compute_layer_hypothesis_rates,
    get_biword_score_pairs_contrast,
    compute_layer_contrast_means,
    pool_head_hypothesis_rates,
    pool_layer_hypothesis_rates,
    pool_biword_score_pairs_contrast,
    pool_layer_contrast_means,
    compute_macro_head_hypothesis_rates,
    compute_macro_layer_hypothesis_rates,
    get_macro_biword_score_pairs_contrast,
    compute_macro_layer_contrast_means,
    pool_macro_head_hypothesis_rates,
    pool_macro_layer_hypothesis_rates,
    pool_macro_biword_score_pairs_contrast,
    pool_macro_layer_contrast_means,
    get_words_by_filter,
    generate_filter_stats,
)
from visualizations import (
    plot_diff_contrast_heatmap,
    plot_layer_contrast_bar,
    plot_per_layer_box_whisker,
    plot_hypothesis_rate_heatmap,
    plot_layer_hypothesis_bar,
    combine_layer_plots,
)
from utils import load_json, load_output_npz, WORD_CATEGORIES


DEFAULT_JSON_PATH = "output/multi_word_output.json"
DEFAULT_THRESHOLD = 0.5
VALID_FILTERS = sorted(set(WORD_CATEGORIES.keys()) | {"other"})


def resolve_json_path(args) -> tuple[str, bool]:
    """
    Returns (json_path, is_temp) where json_path is a path to a usable JSON
    file and is_temp indicates whether it is a temporary file that should be
    deleted after use.

    If --npz is given, loads the npz folder with load_output_npz, serialises
    the result to a temporary JSON file, and returns that path.
    Otherwise returns the normal json_path unchanged.
    """
    if args.npz:
        data = load_output_npz(args.npz)

        for word_data in data["main_data"].values():
            for occ in word_data["occurrences"]:
                arr = occ["attentions"]  # shape: (num_layers, num_heads, seq_len, seq_len)
                occ["attentions"] = {
                    "layers": [
                        {"heads": arr[l].tolist()}
                        for l in range(arr.shape[0])
                    ]
                }

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(data, tmp, ensure_ascii=False)
        tmp.close()
        return tmp.name, True

    return args.json_path, False


def get_dataset_slug(json_path: str, npz_path: str = None) -> str:
    """
    Derives a clean component folder name from the JSON filename or npz folder.
    Always strips any trailing token-count suffix (e.g. _500tokens, _16000tokens).
    """
    import re
    if npz_path:
        return Path(npz_path).name
    slug = Path(json_path).stem.replace("_output", "")
    slug = re.sub(r'_\d+tokens?$', '', slug)
    return slug


def get_base_dir(token_folder: str, component_slug: str, label: str, subfolder: str = None) -> str:
    """
    Returns the output directory: figures/{token_folder}/{component_slug}/[{subfolder}/]{label}/
    Creates it if it doesn't exist.
    """
    parts = [token_folder, component_slug]
    if subfolder:
        parts.append(subfolder)
    parts.append(label)
    base = Path("figures").joinpath(*parts)
    return str(base)


def run_exp2(
    json_path: str,
    output_dir: str,
    nrows: int = 8,
    ncols: int = 4,
) -> None:
    """
    Experiment 2: Box-whisker plots of attention scores per subtoken per head.
    Saves one PNG per layer and stitches them into a combined image.
    """
    from visualizations import get_biword_scores
    print("\n=== Experiment 2: Box-Whisker Plots ===")
    num_layers = len(get_biword_scores(json_path))
    plot_per_layer_box_whisker(
        json_path=json_path,
        output_dir=output_dir,
        nrows=nrows,
        ncols=ncols,
    )
    combine_layer_plots(
        input_dir=output_dir,
        ncols=1,
        nrows=num_layers,
        type="boxplot",
    )


def run_exp3(
    json_path: str,
    output_dir: str,
    nrows: int = 8,
    ncols: int = 4,
) -> None:
    """
    Experiment 3: Subtoken vector polar coordinates (default k0 vs k1).
    Loads text and token indices from the JSON, runs the TransformerLens bridge
    to extract post-RoPE K (and writes/uses slot files in memory via collect),
    then plots polar grids per layer.
    """
    print("\n=== Experiment 3: Vector Polar Coordinates (k0 vs k1) ===")

    in_map = load_json(json_path)
    text = in_map["text"]
    multi_token_words_map = in_map["main_data"]

    from model import get_bridge
    from qkv_vectors import (
        save_slot_vectors,
        collect_vectors,
        micro_average_vectors,
        compute_polar_per_head,
        plot_polar_grid,
    )

    print("Loading TransformerLens bridge for vector extraction...")
    bridge, tokenizer = get_bridge()

    print("Running run_with_cache...")
    import torch
    import tempfile

    tokens = bridge.to_tokens(text, prepend_bos=False)
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens, prepend_bos=False)

    print("Extracting k0/k1 slot vectors...")
    with tempfile.TemporaryDirectory() as tmp:
        save_slot_vectors(
            cache,
            multi_token_words_map,
            tmp,
            slots=("k0", "k1"),
            n_layers=bridge.cfg.n_layers,
        )
        collected = collect_vectors(
            os.path.join(tmp, "k0.pt"),
            os.path.join(tmp, "k1.pt"),
            "k0",
            "k1",
        )
        key_vectors = micro_average_vectors(collected)

    del bridge, cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Freed CUDA memory.")

    print("Computing polar coordinates per head...")
    polar_data = compute_polar_per_head(key_vectors)

    plot_polar_grid(
        polar_data,
        output_dir=output_dir,
        nrows=nrows,
        ncols=ncols,
        title_suffix="Key vector polar coords (k0 vs k1)",
    )
    combine_layer_plots(
        input_dir=output_dir,
        ncols=1,
        nrows=len({layer for layer, _ in polar_data}),
        type="polar_grid",
    )


def filter_json_to_temp(json_path: str, filter_name: str) -> tuple[str, int]:
    """
    Loads the JSON, keeps only words matching filter_name in main_data,
    writes to a temp file, and returns (temp_path, matched_word_count).
    Caller is responsible for deleting the temp file.
    """
    data = load_json(json_path)
    matched_words = get_words_by_filter(json_path, filter_name)
    filtered_main = {w: data["main_data"][w] for w in matched_words}
    filtered_data = {**data, "main_data": filtered_main}

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(filtered_data, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name, len(matched_words)


def _build_context(
    json_path: str,
    word: str = None,
    label: str = None,
    averaging: str = None,
    n_words: int = None,
    n_occurrences: int = None,
) -> str:
    """
    Builds a human-readable context string for plot titles from JSON metadata.
    E.g.: "Enron Emails | 500 tokens | word: '2000' | micro | 1 word, 31 occurrences"
    """
    try:
        meta = load_json(json_path)
        component = meta.get("component", Path(json_path).stem)
        num_tokens = meta.get("num_tokens")
    except Exception:
        component = Path(json_path).stem
        num_tokens = None

    parts = [component]
    if num_tokens:
        parts.append(f"{num_tokens} tokens")
    if word is not None:
        parts.append(f"word: {repr(word)}")
    elif label and label != "all":
        parts.append(f"group: {label}")
    if averaging:
        parts.append(averaging)
    if n_words is not None and n_occurrences is not None:
        parts.append(f"{n_words} word{'s' if n_words != 1 else ''}, {n_occurrences} occurrence{'s' if n_occurrences != 1 else ''}")
    return " | ".join(parts)


def _sample_size(json_path: str, word: str = None) -> tuple[int, int]:
    """Returns (n_words, n_occurrences) for the given JSON, optionally filtered to one word."""
    mw_map = load_json(json_path)["main_data"]
    if word is not None:
        subset = {word: mw_map[word]} if word in mw_map else {}
    else:
        subset = mw_map
    n_words = len(subset)
    n_occurrences = sum(len(info["occurrences"]) for info in subset.values())
    return n_words, n_occurrences


def run_exp4(
    json_path: str,
    output_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
    word: str = None,
    label: str = "all",
) -> None:
    """
    Experiment 4: Hypothesis rate analysis (heatmap + per-layer bar chart).
    Plots both micro-average (all occurrences) and macro-average (equal weight per word type).

    Args:
        word:  if provided, restrict analysis to this single word's occurrences only
        label: subfolder label, used for plot title context
    """
    print("\n=== Experiment 4: Hypothesis Rate Analysis ===")
    if word:
        print(f"    Word filter: {repr(word)}")

    n_words, n_occurrences = _sample_size(json_path, word=word)
    context_micro = _build_context(json_path, word=word, label=label, averaging="micro", n_words=n_words, n_occurrences=n_occurrences)
    context_macro = _build_context(json_path, word=word, label=label, averaging="macro", n_words=n_words, n_occurrences=n_occurrences)

    rates_head_micro = compute_head_hypothesis_rates(json_path, word=word)
    rates_layer_micro = compute_layer_hypothesis_rates(json_path, word=word)
    passing_micro = sum(1 for v in rates_head_micro.values() if v >= threshold)
    print(f"Micro — Threshold: {threshold:.0%} | "
          f"{passing_micro}/{len(rates_head_micro)} heads pass ({100*passing_micro/len(rates_head_micro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_micro, output_dir=output_dir, suffix="_micro", context=context_micro)
    plot_layer_hypothesis_bar(rates_layer_micro, output_dir=output_dir, threshold=threshold, suffix="_micro", context=context_micro)

    rates_head_macro = compute_macro_head_hypothesis_rates(json_path, word=word)
    rates_layer_macro = compute_macro_layer_hypothesis_rates(json_path, word=word)
    passing_macro = sum(1 for v in rates_head_macro.values() if v >= threshold)
    print(f"Macro — Threshold: {threshold:.0%} | "
          f"{passing_macro}/{len(rates_head_macro)} heads pass ({100*passing_macro/len(rates_head_macro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_macro, output_dir=output_dir, suffix="_macro", context=context_macro)
    plot_layer_hypothesis_bar(rates_layer_macro, output_dir=output_dir, threshold=threshold, suffix="_macro", context=context_macro)


def run_exp5(
    json_path: str,
    output_dir: str,
    word: str = None,
    label: str = "all",
) -> None:
    """
    Experiment 5: Michelson contrast analysis (heatmap + per-layer bar chart).
    Plots both micro-average (all occurrences) and macro-average (equal weight per word type).

    Args:
        word:  if provided, restrict analysis to this single word's occurrences only
        label: subfolder label, used for plot title context
    """
    print("\n=== Experiment 5: Michelson Contrast Analysis ===")
    if word:
        print(f"    Word filter: {repr(word)}")

    n_words, n_occurrences = _sample_size(json_path, word=word)
    context_micro = _build_context(json_path, word=word, label=label, averaging="micro", n_words=n_words, n_occurrences=n_occurrences)
    context_macro = _build_context(json_path, word=word, label=label, averaging="macro", n_words=n_words, n_occurrences=n_occurrences)

    contrasts_micro = get_biword_score_pairs_contrast(json_path, word=word)
    layer_contrasts_micro = compute_layer_contrast_means(json_path, word=word)
    print(f"Micro — Loaded {sum(len(v) for v in contrasts_micro.values())} contrast values "
          f"across {len(contrasts_micro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_micro, output_dir=output_dir, suffix="_micro", context=context_micro)
    plot_layer_contrast_bar(layer_contrasts_micro, output_dir=output_dir, suffix="_micro", context=context_micro)

    contrasts_macro = get_macro_biword_score_pairs_contrast(json_path, word=word)
    layer_contrasts_macro = compute_macro_layer_contrast_means(json_path, word=word)
    print(f"Macro — Loaded {sum(len(v) for v in contrasts_macro.values())} contrast values "
          f"across {len(contrasts_macro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_macro, output_dir=output_dir, suffix="_macro", context=context_macro)
    plot_layer_contrast_bar(layer_contrasts_macro, output_dir=output_dir, suffix="_macro", context=context_macro)


def run_exp6(json_paths: list, output_dir: str, threshold: float = DEFAULT_THRESHOLD) -> None:
    """
    Experiment 6: Pooled hypothesis rate analysis (heatmap + per-layer bar chart).
    Pools raw pairs from all component JSONs before computing micro and macro rates.
    """
    print("\n=== Experiment 6: Pooled Hypothesis Rate Analysis ===")

    rates_head_micro = pool_head_hypothesis_rates(json_paths)
    rates_layer_micro = pool_layer_hypothesis_rates(json_paths)
    passing_micro = sum(1 for v in rates_head_micro.values() if v >= threshold)
    print(f"Micro — Threshold: {threshold:.0%} | "
          f"{passing_micro}/{len(rates_head_micro)} heads pass ({100*passing_micro/len(rates_head_micro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_micro, output_dir=output_dir, suffix="_micro")
    plot_layer_hypothesis_bar(rates_layer_micro, output_dir=output_dir, threshold=threshold, suffix="_micro")

    rates_head_macro = pool_macro_head_hypothesis_rates(json_paths)
    rates_layer_macro = pool_macro_layer_hypothesis_rates(json_paths)
    passing_macro = sum(1 for v in rates_head_macro.values() if v >= threshold)
    print(f"Macro — Threshold: {threshold:.0%} | "
          f"{passing_macro}/{len(rates_head_macro)} heads pass ({100*passing_macro/len(rates_head_macro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_macro, output_dir=output_dir, suffix="_macro")
    plot_layer_hypothesis_bar(rates_layer_macro, output_dir=output_dir, threshold=threshold, suffix="_macro")


def run_exp7(json_paths: list, output_dir: str) -> None:
    """
    Experiment 7: Pooled Michelson contrast analysis (heatmap + per-layer bar chart).
    Pools raw pairs from all component JSONs before computing micro and macro contrast metrics.
    """
    print("\n=== Experiment 7: Pooled Michelson Contrast Analysis ===")

    contrasts_micro = pool_biword_score_pairs_contrast(json_paths)
    layer_contrasts_micro = pool_layer_contrast_means(json_paths)
    print(f"Micro — Pooled {sum(len(v) for v in contrasts_micro.values())} contrast values "
          f"across {len(contrasts_micro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_micro, output_dir=output_dir, suffix="_micro")
    plot_layer_contrast_bar(layer_contrasts_micro, output_dir=output_dir, suffix="_micro")

    contrasts_macro = pool_macro_biword_score_pairs_contrast(json_paths)
    layer_contrasts_macro = pool_macro_layer_contrast_means(json_paths)
    print(f"Macro — Pooled {sum(len(v) for v in contrasts_macro.values())} contrast values "
          f"across {len(contrasts_macro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_macro, output_dir=output_dir, suffix="_macro")
    plot_layer_contrast_bar(layer_contrasts_macro, output_dir=output_dir, suffix="_macro")


def get_pt_slug(pt_path: str) -> str:
    """
    Derives a component slug from a slot directory or legacy .pt filename.
    E.g. Pile-CC_16000tokens → Pile-CC
         Pile-CC_16000tokens_kv.pt → Pile-CC
    """
    import re
    stem = Path(pt_path).stem if Path(pt_path).is_file() else Path(pt_path).name
    stem = re.sub(r'_\d+tokens_kv$', '', stem)
    stem = re.sub(r'_\d+tokens$', '', stem)
    return stem


def _slot_dir_has_pair(slot_dir: Path, name_a: str = "k0", name_b: str = "k1") -> bool:
    return (slot_dir / f"{name_a}.pt").is_file() and (slot_dir / f"{name_b}.pt").is_file()


def _slot_dir_has_all_slots(slot_dir: Path) -> bool:
    from qkv_vectors import SLOTS
    return all((slot_dir / f"{s}.pt").is_file() for s in SLOTS)


def run_exp8(
    slot_dir: str,
    base_dir: str,
    name_a: str = "k0",
    name_b: str = "k1",
) -> None:
    """
    Experiment 8: one QKV slot-pair polar heatmaps.
    Loads two slot .pt files from a component slot directory and produces r/theta
    heatmaps for each word category using micro- and macro-averaging.
    No model re-run required.

    Polar convention: r = ||name_a|| / ||name_b|| (name_b is the reference).
    Prefer order_slot_pair() so name_a is earlier subtoken and name_b is later.

    Args:
        slot_dir: directory containing {name_a}.pt and {name_b}.pt from save_slot_vectors
        base_dir: output root — filter subfolders are written directly under here
        name_a / name_b: which slots to pair (default k0 vs k1)

    Output per filter: {base_dir}/<filter>/
        kv_theta_heatmap_micro.png
        kv_theta_heatmap_macro.png
        kv_r_heatmap_micro.png
        kv_r_heatmap_macro.png
    """
    print(f"\n=== Experiment 8: Polar Heatmaps ({name_a} vs {name_b}) ===")
    print(f"  r = ||{name_a}|| / ||{name_b}||")

    from qkv_vectors import (
        collect_vectors,
        filter_collected,
        micro_average_vectors,
        macro_average_vectors,
        compute_polar_per_head,
    )
    from visualizations import plot_polar_heatmaps

    path_a = os.path.join(slot_dir, f"{name_a}.pt")
    path_b = os.path.join(slot_dir, f"{name_b}.pt")
    if not os.path.isfile(path_a) or not os.path.isfile(path_b):
        print(f"[exp8] skip {name_a} vs {name_b}: missing {path_a} or {path_b}")
        return

    collected = collect_vectors(path_a, path_b, name_a, name_b)
    print(f"[exp8] Loaded {len(collected)} words from {slot_dir} ({name_a} vs {name_b})")

    filters = ["all", "space_numbers", "numbers", "words", "space_words"]
    for filter_name in filters:
        filtered = filter_collected(collected, filter_name)
        if not filtered:
            print(f"[exp8] skip '{filter_name}': no words matched")
            continue
        n_words = len(filtered)
        n_occurrences = sum(
            len(next(iter(wd.values()))[name_a]) for wd in filtered.values()
        )
        print(f"[exp8] Filter '{filter_name}': {n_words} words, {n_occurrences} occurrences")
        out_dir = os.path.join(base_dir, filter_name)
        for avg_name, avg_fn in [("micro", micro_average_vectors),
                                  ("macro", macro_average_vectors)]:
            averaged = avg_fn(filtered)
            polar_data = compute_polar_per_head(averaged, pair=(name_a, name_b))
            plot_polar_heatmaps(
                polar_data,
                output_dir=out_dir,
                suffix=f"_{avg_name}",
                context=(
                    f"{filter_name} | {avg_name}-avg | {Path(slot_dir).name} | "
                    f"{n_words} word{'s' if n_words != 1 else ''}, "
                    f"{n_occurrences} occurrence{'s' if n_occurrences != 1 else ''} | "
                    f"r = ||{name_a}|| / ||{name_b}||"
                ),
                name_a=name_a,
                name_b=name_b,
            )


def run_all_qkv_pair_polar_heatmaps(slot_dir: str, out_root: str) -> None:
    """
    Polar heatmaps for every ordered C(6,2)=15 QKV slot pair.

    Pair-order rule (see qkv_vectors.order_slot_pair):
      - A = earlier subtoken index, B = later (only (0,0), (0,1), (1,1))
      - same-index ties broken by role q < k < v
      - r = ||A|| / ||B|| with B as reference

    Output:
      {out_root}/{name_a}_vs_{name_b}/<filter>/kv_{r,theta}_heatmap_{micro,macro}.png
    """
    from qkv_vectors import SLOTS, iter_ordered_slot_pairs

    slot_dir_p = Path(slot_dir)
    missing = [s for s in SLOTS if not (slot_dir_p / f"{s}.pt").is_file()]
    if missing:
        print(f"[exp8-all] Warning: missing slot files in {slot_dir}: {missing} "
              f"(pairs that need them will be skipped)")

    pairs = iter_ordered_slot_pairs()
    print(f"\n=== All QKV pair polar heatmaps ({len(pairs)} pairs) ===")
    print(f"  slot_dir: {slot_dir}")
    print(f"  out_root: {out_root}")

    for name_a, name_b in pairs:
        pair_dir = os.path.join(out_root, f"{name_a}_vs_{name_b}")
        run_exp8(slot_dir, pair_dir, name_a=name_a, name_b=name_b)


def _run_all_exps(json_path: str, attn_base_dir: str, other_base_dir: str, args) -> None:
    """Run all selected experiments for a single JSON file.
    Attention experiments (2, 4, 5) write to attn_base_dir.
    Exp 3 writes to other_base_dir.
    """
    word = getattr(args, "word", None)
    label = getattr(args, "label", "all")
    filter_name = getattr(args, "filter", None)

    temp_path = None
    try:
        if filter_name is not None:
            temp_path, n_matched = filter_json_to_temp(json_path, filter_name)
            if n_matched == 0:
                print(f"\n[Warning] No words matched filter {repr(filter_name)} in {json_path}. Skipping.")
                return
            print(f"  Filter {repr(filter_name)}: {n_matched} words matched.")
            json_path = temp_path

        if word is not None:
            available = load_json(json_path)["main_data"].keys()
            if word not in available:
                available_sorted = sorted(available)
                preview = available_sorted[:20]
                print(f"\n[Error] Word {repr(word)} not found in {json_path}.")
                print(f"  {len(available_sorted)} words available. First 20: {preview}")
                if len(available_sorted) > 20:
                    print(f"  ... and {len(available_sorted) - 20} more.")
                print("  Tip: use rank_words_by_occurrence() to see all words ranked by frequency.")
                return

        if 2 in args.exp:
            run_exp2(
                json_path,
                output_dir=attn_base_dir,
                nrows=args.nrows,
                ncols=args.ncols,
            )

        if 3 in args.exp:
            run_exp3(
                json_path,
                output_dir=other_base_dir,
                nrows=args.nrows,
                ncols=args.ncols,
            )

        if 4 in args.exp:
            run_exp4(
                json_path,
                output_dir=attn_base_dir,
                threshold=args.threshold,
                word=word,
                label=label,
            )

        if 5 in args.exp:
            run_exp5(
                json_path,
                output_dir=attn_base_dir,
                word=word,
                label=label,
            )

        print(f"\nAll selected experiments complete. Attention outputs in {attn_base_dir}/")
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Run attention hypothesis experiments.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help=f"Path to the output JSON from main.py (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Path to a folder of per-component JSONs (e.g. output/500_tokens/). "
             "Runs experiments separately for each .json file found. "
             "If given, json_path and --npz are ignored.",
    )
    parser.add_argument(
        "--npz",
        default=None,
        help="Path to an npz output folder created by save_output_npz. "
             "If given, json_path is ignored.",
    )
    parser.add_argument(
        "--pt",
        default=None,
        help="Path to a QKV slot directory from the kv/qkv run (required for exp 8). "
             "E.g. output/qkv_cache/16000_tokens/Pile-CC_16000tokens "
             "or a parent dir containing such component folders.",
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        type=int,
        choices=[2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Which experiments to run (default: all unless --stats is the only flag). "
             "Exps 6-7 are pooled versions and only run in --folder mode. "
             "Exp 8 requires --pt.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Hypothesis rate threshold for Exp 4 and 6 (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--nrows", type=int, default=8, help="Subplot rows for plot grids (default: 8)"
    )
    parser.add_argument(
        "--ncols", type=int, default=4, help="Subplot cols for plot grids (default: 4)"
    )
    parser.add_argument(
        "--word",
        default=None,
        help="Restrict exp 4 and 5 to a single word's occurrences (e.g. --word '2000'). "
             "Use rank_words_by_occurrence() to see available words in a JSON.",
    )
    parser.add_argument(
        "--label",
        default="all",
        help="Subfolder label under figures/{tokens}/{component}/. "
             "Controls where outputs are saved (default: 'all'). "
             "Examples: 'all', 'word_2000', 'numbers', 'symbols'.",
    )
    parser.add_argument(
        "--filter",
        default=None,
        choices=VALID_FILTERS,
        help=f"Filter words by category before running exp 4/5. "
             f"Valid options: {', '.join(VALID_FILTERS)}",
    )
    parser.add_argument(
        "--all-filters",
        action="store_true",
        default=False,
        help="Run exp 4/5 once per filter category. Ignores --filter and --label.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        default=False,
        help="Generate a filter stats table (words/occurrences per category) and save "
             "it alongside the figures. Can be used alone or with --exp.",
    )

    args = parser.parse_args()

    if args.exp is None:
        args.exp = [] if args.stats and not args.all_filters else [2, 3, 4, 5, 6, 7]

    if args.all_filters:
        for filter_name in ["all"] + VALID_FILTERS:
            print(f"\n{'='*60}")
            print(f"  Running filter: {filter_name}")
            print(f"{'='*60}")
            filter_args = argparse.Namespace(**vars(args))
            filter_args.all_filters = False
            if filter_name == "all":
                filter_args.filter = None
                filter_args.label = "all"
            else:
                filter_args.filter = filter_name
                filter_args.label = filter_name
            _main_run(filter_args)
        return

    if args.word and args.label == "all":
        args.label = f"word_{args.word}"
    elif args.filter and args.label == "all":
        args.label = args.filter

    _main_run(args)


def _write_stats(json_path: str, base_dir: str) -> None:
    """Generate and save a filter stats table to the component directory."""
    stats_text = generate_filter_stats(json_path)
    # base_dir is figures/{tokens}/{component}/attention/{label}
    # so parent.parent is the component dir
    component_dir = Path(base_dir).parent.parent
    component_dir.mkdir(parents=True, exist_ok=True)
    out_path = component_dir / "filter_stats.txt"
    out_path.write_text(stats_text, encoding="utf-8")
    print(f"Saved {out_path}")
    print(stats_text)


def _main_run(args) -> None:
    if args.folder:
        json_files = sorted(Path(args.folder).glob("*.json"))
        if not json_files:
            print(f"No .json files found in {args.folder}")
            return
        print(f"Folder mode: found {len(json_files)} JSON file(s) in {args.folder}")
        print(f"Experiments to run: {args.exp}\n")
        folder_name = Path(args.folder).name
        json_paths = [str(f) for f in json_files]
        for json_file in json_files:
            json_path = str(json_file)
            slug = get_dataset_slug(json_path)
            attn_base = get_base_dir(folder_name, slug, args.label, subfolder="attention")
            other_base = get_base_dir(folder_name, slug, args.label, subfolder="other")
            print(f"--- Input: {json_path}")
            print(f"    Attention output dir: {attn_base}/")
            if getattr(args, "stats", False):
                _write_stats(json_path, attn_base)
            _run_all_exps(json_path, attn_base, other_base, args)

        pooled_exps = [e for e in [6, 7] if e in args.exp]
        if pooled_exps:
            print(f"\n--- Pooled experiments {pooled_exps} across {len(json_paths)} components ---")
            pooled_base = get_base_dir(folder_name, "_pooled", args.label, subfolder="attention")
            if 6 in args.exp:
                run_exp6(json_paths, output_dir=pooled_base, threshold=args.threshold)
            if 7 in args.exp:
                run_exp7(json_paths, output_dir=pooled_base)
            print(f"\nPooled outputs in {pooled_base}/")
    else:
        is_temp = False
        try:
            json_path, is_temp = resolve_json_path(args)
            token_folder = Path(json_path).parent.name
            slug = get_dataset_slug(json_path, npz_path=args.npz)
            attn_base = get_base_dir(token_folder, slug, args.label, subfolder="attention")
            other_base = get_base_dir(token_folder, slug, args.label, subfolder="other")

            source_label = args.npz if args.npz else json_path
            print(f"Input:               {source_label}")
            print(f"Dataset slug:        {slug}")
            print(f"Attention output:    {attn_base}/")
            print(f"Experiments to run:  {args.exp}")

            if getattr(args, "stats", False):
                _write_stats(json_path, attn_base)
            _run_all_exps(json_path, attn_base, other_base, args)
        finally:
            if is_temp:
                Path(json_path).unlink(missing_ok=True)

    if 8 in args.exp:
        if not args.pt:
            print("[exp8] Skipping: --pt not provided. Pass a slot directory "
                  "(with q0..v1 .pt files) or a parent of such directories.")
        elif os.path.isdir(args.pt):
            pt_root = Path(args.pt)
            # Prefer dirs that have the full six slots; fall back to any with k0+k1
            def _is_slot_dir(d: Path) -> bool:
                return _slot_dir_has_all_slots(d) or _slot_dir_has_pair(d)

            if _is_slot_dir(pt_root):
                token_folder = pt_root.parent.name
                slug = get_pt_slug(str(pt_root))
                base_dir_8 = get_base_dir(token_folder, slug, "polar")
                run_all_qkv_pair_polar_heatmaps(str(pt_root), base_dir_8)
            else:
                slot_dirs = sorted(
                    d for d in pt_root.iterdir()
                    if d.is_dir() and _is_slot_dir(d)
                )
                if not slot_dirs:
                    print(f"[exp8] No slot directories with QKV .pt files found in {args.pt}")
                else:
                    token_folder = pt_root.name
                    for slot_dir in slot_dirs:
                        slug = get_pt_slug(str(slot_dir))
                        base_dir_8 = get_base_dir(token_folder, slug, "polar")
                        run_all_qkv_pair_polar_heatmaps(str(slot_dir), base_dir_8)
        else:
            print(f"[exp8] --pt must be a directory of slot .pt files, got file: {args.pt}")


if __name__ == "__main__":
    main()
