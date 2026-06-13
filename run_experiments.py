"""
run_experiments.py

Entry point for running all hypothesis experiments against a saved output JSON
(produced by main.py). Does not re-run the model except for exp3 (key vectors).

Usage:
    python run_experiments.py                                    # uses default path
    python run_experiments.py output/my_output.json              # single JSON
    python run_experiments.py output/my_output.json --exp 2 4    # run only exp 2 and 4
    python run_experiments.py --folder output/500_tokens/        # all JSONs in folder
    python run_experiments.py --folder output/500_tokens/ --exp 4 5  # subset of exps

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
)
from visualizations import (
    plot_diff_contrast_heatmap,
    plot_layer_contrast_bar,
    plot_per_layer_box_whisker,
    plot_hypothesis_rate_heatmap,
    plot_layer_hypothesis_bar,
    combine_layer_plots,
)
from utils import load_json, load_output_npz


DEFAULT_JSON_PATH = "output/multi_word_output.json"
DEFAULT_THRESHOLD = 0.5


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


def get_dataset_slug(json_path: str, npz_path: str = None, parent_folder_name: str = None) -> str:
    """
    Derives a clean folder name from the JSON filename or npz folder.
    If parent_folder_name is provided, removes token count suffix (e.g., _500tokens)
    that matches the parent folder name.
    """
    import re
    if npz_path:
        return Path(npz_path).name
    slug = Path(json_path).stem.replace("_output", "")
    if parent_folder_name:
        # Try removing suffix like _500tokens that matches the parent folder's token count
        # e.g., if parent is "500_tokens", remove "_500tokens" from slug
        match = re.search(r'_(\d+)tokens?$', slug)
        if match and match.group(1) in parent_folder_name:
            slug = slug[:match.start()]
    return slug


def get_unique_base_dir(slug: str, use_figures_root: bool = True) -> str:
    """
    Returns the base directory for outputs. If use_figures_root is True, prepends
    'figures/' to the slug (for backward compatibility). Otherwise uses slug as-is.
    Appends (1), (2), etc. if the directory already exists.
    """
    if use_figures_root:
        base = Path("figures") / slug
    else:
        base = Path(slug)

    if not base.exists():
        return str(base)
    counter = 1
    while True:
        if use_figures_root:
            candidate = Path("figures") / f"{slug} ({counter})"
        else:
            parent = base.parent
            name = base.name
            candidate = parent / f"{name} ({counter})"
        if not candidate.exists():
            return str(candidate)
        counter += 1


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


def run_exp4(
    json_path: str,
    output_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Experiment 4: Hypothesis rate analysis (heatmap + per-layer bar chart).
    Plots both micro-average (all occurrences) and macro-average (equal weight per word type).
    """
    print("\n=== Experiment 4: Hypothesis Rate Analysis ===")

    rates_head_micro = compute_head_hypothesis_rates(json_path)
    rates_layer_micro = compute_layer_hypothesis_rates(json_path)
    passing_micro = sum(1 for v in rates_head_micro.values() if v >= threshold)
    print(f"Micro — Threshold: {threshold:.0%} | "
          f"{passing_micro}/{len(rates_head_micro)} heads pass ({100*passing_micro/len(rates_head_micro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_micro, output_dir=output_dir, suffix="_micro")
    plot_layer_hypothesis_bar(rates_layer_micro, output_dir=output_dir, threshold=threshold, suffix="_micro")

    rates_head_macro = compute_macro_head_hypothesis_rates(json_path)
    rates_layer_macro = compute_macro_layer_hypothesis_rates(json_path)
    passing_macro = sum(1 for v in rates_head_macro.values() if v >= threshold)
    print(f"Macro — Threshold: {threshold:.0%} | "
          f"{passing_macro}/{len(rates_head_macro)} heads pass ({100*passing_macro/len(rates_head_macro):.1f}%)")
    plot_hypothesis_rate_heatmap(rates_head_macro, output_dir=output_dir, suffix="_macro")
    plot_layer_hypothesis_bar(rates_layer_macro, output_dir=output_dir, threshold=threshold, suffix="_macro")


def run_exp5(
    json_path: str,
    output_dir: str,
) -> None:
    """
    Experiment 5: Michelson contrast analysis (heatmap + per-layer bar chart).
    Plots both micro-average (all occurrences) and macro-average (equal weight per word type).
    """
    print("\n=== Experiment 5: Michelson Contrast Analysis ===")

    contrasts_micro = get_biword_score_pairs_contrast(json_path)
    layer_contrasts_micro = compute_layer_contrast_means(json_path)
    print(f"Micro — Loaded {sum(len(v) for v in contrasts_micro.values())} contrast values "
          f"across {len(contrasts_micro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_micro, output_dir=output_dir, suffix="_micro")
    plot_layer_contrast_bar(layer_contrasts_micro, output_dir=output_dir, suffix="_micro")

    contrasts_macro = get_macro_biword_score_pairs_contrast(json_path)
    layer_contrasts_macro = compute_macro_layer_contrast_means(json_path)
    print(f"Macro — Loaded {sum(len(v) for v in contrasts_macro.values())} contrast values "
          f"across {len(contrasts_macro)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts_macro, output_dir=output_dir, suffix="_macro")
    plot_layer_contrast_bar(layer_contrasts_macro, output_dir=output_dir, suffix="_macro")


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
        )

    if 5 in args.exp:
        run_exp5(
            json_path,
            output_dir=base_dir,
        )

    print(f"\nAll selected experiments complete. Outputs in {base_dir}/")


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
        default=[2, 3, 4, 5, 6, 7],
        help="Which experiments to run (default: all). "
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

    args = parser.parse_args()

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
            base_dir = get_unique_base_dir(f"{folder_name}/{slug}")
            print(f"--- Input: {json_path}")
            print(f"    Output base dir: {base_dir}/")
            _run_all_exps(json_path, base_dir, args)

        pooled_exps = [e for e in [6, 7] if e in args.exp]
        if pooled_exps:
            print(f"\n--- Pooled experiments {pooled_exps} across {len(json_paths)} components ---")
            pooled_base = get_unique_base_dir(f"{folder_name}/_pooled")
            if 6 in args.exp:
                run_exp6(json_paths, output_dir=pooled_base, threshold=args.threshold)
            if 7 in args.exp:
                run_exp7(json_paths, output_dir=pooled_base)
            print(f"\nPooled outputs in {pooled_base}/")
    else:
        is_temp = False
        try:
            json_path, is_temp = resolve_json_path(args)
            parent_folder = Path(json_path).parent.name
            slug = get_dataset_slug(json_path, npz_path=args.npz, parent_folder_name=parent_folder)
            base_dir = get_unique_base_dir(f"{parent_folder}/{slug}")

            source_label = args.npz if args.npz else json_path
            print(f"Input:               {source_label}")
            print(f"Dataset slug:        {slug}")
            print(f"Output base dir:     {base_dir}/")
            print(f"Experiments to run:  {args.exp}")

            _run_all_exps(json_path, base_dir, args)
        finally:
            if is_temp:
                Path(json_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
