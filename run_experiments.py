"""
run_experiments.py

Entry point for running all hypothesis experiments against a saved output JSON
(produced by main.py). Does not re-run the model except for exp3 (key vectors).

Usage:
    python run_experiments.py                                             # uses default path
    python run_experiments.py output/my_output.json                       # single JSON
    python run_experiments.py output/my_output.json --exp 2 4             # run only exp 2 and 4
    python run_experiments.py --folder output/500_tokens/                 # all JSONs in folder
    python run_experiments.py --folder output/500_tokens/ --exp 4 5       # subset of exps
    python run_experiments.py output/my_output.json --exp 4 5 --word 'Colo'           # single word
    python run_experiments.py output/my_output.json --exp 4 5 --filter space_numbers  # word category

Experiments:
    2 — Box-whisker plots: attention score distributions per subtoken per head
    3 — Key vector polar coordinates: geometric comparison of subtoken key vectors
        (requires re-running the model to extract past_key_values)
    4 — Hypothesis rate analysis: heatmap + per-layer bar chart
    5 — Michelson contrast analysis: heatmap + per-layer bar chart

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
"""

import argparse
import json
import tempfile
from pathlib import Path

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


def get_base_dir(token_folder: str, component_slug: str, label: str) -> str:
    """
    Returns the output directory: figures/{token_folder}/{component_slug}/{label}/
    Creates it if it doesn't exist.
    """
    base = Path("figures") / token_folder / component_slug / label
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
    Experiment 3: Key vector polar coordinates.
    Loads the original text and token indices from the JSON, re-runs the
    model to get past_key_values, then plots polar grids per layer.
    Token indices come directly from the JSON — no re-tokenization needed.
    """
    print("\n=== Experiment 3: Key Vector Polar Coordinates ===")

    in_map = load_json(json_path)
    text = in_map["text"]
    multi_token_words_map = in_map["main_data"]

    from model import get_model
    from keys import extract_key_vectors, compute_polar_per_head, plot_polar_grid

    print("Loading model for key vector extraction...")
    model, tokenizer = get_model()

    print("Extracting key vectors...")
    key_vectors = extract_key_vectors(text, model, tokenizer, multi_token_words_map)

    del model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Freed CUDA memory.")

    print("Computing polar coordinates per head...")
    polar_data = compute_polar_per_head(key_vectors)

    plot_polar_grid(polar_data, output_dir=output_dir, nrows=nrows, ncols=ncols)
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



def _run_all_exps(json_path: str, base_dir: str, args) -> None:
    """Run all selected experiments for a single JSON file, writing into base_dir."""
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
                output_dir=base_dir,
                nrows=args.nrows,
                ncols=args.ncols,
            )

        if 3 in args.exp:
            run_exp3(
                json_path,
                output_dir=base_dir,
                nrows=args.nrows,
                ncols=args.ncols,
            )

        if 4 in args.exp:
            run_exp4(
                json_path,
                output_dir=base_dir,
                threshold=args.threshold,
                word=word,
                label=label,
            )

        if 5 in args.exp:
            run_exp5(
                json_path,
                output_dir=base_dir,
                word=word,
                label=label,
            )

        print(f"\nAll selected experiments complete. Outputs in {base_dir}/")
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
        "--exp",
        nargs="+",
        type=int,
        choices=[2, 3, 4, 5, 6, 7],
        default=None,
        help="Which experiments to run (default: all unless --stats is the only flag). "
             "Exps 6-7 are pooled versions and only run in --folder mode.",
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
    """Generate and save a filter stats table to the component parent directory."""
    stats_text = generate_filter_stats(json_path)
    component_dir = Path(base_dir).parent
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
            base_dir = get_base_dir(folder_name, slug, args.label)
            print(f"--- Input: {json_path}")
            print(f"    Output base dir: {base_dir}/")
            if getattr(args, "stats", False):
                _write_stats(json_path, base_dir)
            _run_all_exps(json_path, base_dir, args)

        pooled_exps = [e for e in [6, 7] if e in args.exp]
        if pooled_exps:
            print(f"\n--- Pooled experiments {pooled_exps} across {len(json_paths)} components ---")
            pooled_base = get_base_dir(folder_name, "_pooled", args.label)
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
            base_dir = get_base_dir(token_folder, slug, args.label)

            source_label = args.npz if args.npz else json_path
            print(f"Input:               {source_label}")
            print(f"Dataset slug:        {slug}")
            print(f"Output base dir:     {base_dir}/")
            print(f"Experiments to run:  {args.exp}")

            if getattr(args, "stats", False):
                _write_stats(json_path, base_dir)
            _run_all_exps(json_path, base_dir, args)
        finally:
            if is_temp:
                Path(json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
