"""
run_experiments.py

Entry point for running all hypothesis experiments against a saved output JSON
(produced by main.py). Does not re-run the model except for exp3 (key vectors).

Usage:
    python run_experiments.py                                  # uses default path
    python run_experiments.py output/my_output.json            # custom path
    python run_experiments.py output/my_output.json --exp 1 3  # run only exp 1 and 3

Experiments:
    1 — Pairwise difference histograms: tok_1 - tok_0 per row, per head
    2 — Box-whisker plots: attention score distributions per subtoken per head
    3 — Key vector polar coordinates: geometric comparison of subtoken key vectors
        (requires re-running the model to extract past_key_values)
    4 — Hypothesis rate heatmap: 2D grid of (layer x head) hypothesis rates

Output folder structure:
    visuals/<dataset_slug>/
        diff_histograms/
        boxplots/
        polar/
        heatmap/

    The dataset_slug is derived from the JSON filename, e.g.:
        output/10_paragraphs_2-token_max_word_output.json
        → visuals/10_paragraphs_2-token_max_word/
"""

import argparse
from pathlib import Path

from analysis import (
    compute_head_hypothesis_rates,
    get_biword_score_pairs_diff,
)
from visualizations import (
    plot_difference_histograms,
    plot_per_layer_box_whisker,
    plot_hypothesis_rate_heatmap,
    combine_layer_plots,
)
from utils import load_json


DEFAULT_JSON_PATH = "output/multi_word_output.json"
DEFAULT_THRESHOLD = 0.5


def get_dataset_slug(json_path: str) -> str:
    """Derives a clean folder name from the JSON filename."""
    return Path(json_path).stem.replace("_output", "")


def get_unique_base_dir(slug: str) -> str:
    """
    Returns visuals/<slug>/ if it doesn't exist, otherwise
    visuals/<slug> (1)/, visuals/<slug> (2)/, etc.
    """
    base = Path("visuals") / slug
    if not base.exists():
        return str(base)
    counter = 1
    while True:
        candidate = Path("visuals") / f"{slug} ({counter})"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def run_exp1(
    json_path: str,
    output_dir: str,
    nrows: int = 8,
    ncols: int = 4,
) -> None:
    """
    Experiment 1: Pairwise difference histograms (tok_1 - tok_0 per row).
    Saves one PNG per layer and stitches them into a combined image.
    """
    print("\n=== Experiment 1: Pairwise Difference Histograms ===")
    diffs = get_biword_score_pairs_diff(json_path)
    print(f"Loaded {sum(len(v) for v in diffs.values())} paired differences "
          f"across {len(diffs)} (layer, head) pairs.")
    plot_difference_histograms(diffs, output_dir=output_dir, nrows=nrows, ncols=ncols)
    combine_layer_plots(
        input_dir=output_dir,
        ncols=1,
        nrows=len({layer for layer, _ in diffs}),
        type="diff_histogram",
    )


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
    Experiment 4: Hypothesis rate heatmap.
    Plots a 2D (layer x head) heatmap of the fraction of row-level attention
    comparisons where tok_1 > tok_0.
    """
    print("\n=== Experiment 4: Hypothesis Rate Heatmap ===")
    rates = compute_head_hypothesis_rates(json_path)
    passing = sum(1 for v in rates.values() if v >= threshold)
    print(f"Threshold: {threshold:.0%} | "
          f"{passing}/{len(rates)} heads pass ({100*passing/len(rates):.1f}%)")
    plot_hypothesis_rate_heatmap(rates, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run attention hypothesis experiments.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=DEFAULT_JSON_PATH,
        help=f"Path to the output JSON from main.py (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--exp",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="Which experiments to run (default: all). E.g. --exp 1 4",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Hypothesis rate threshold for Exp 4 heatmap (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--nrows", type=int, default=8, help="Subplot rows for plot grids (default: 8)"
    )
    parser.add_argument(
        "--ncols", type=int, default=4, help="Subplot cols for plot grids (default: 4)"
    )

    args = parser.parse_args()

    slug = get_dataset_slug(args.json_path)
    base_dir = get_unique_base_dir(slug)

    print(f"JSON path:           {args.json_path}")
    print(f"Dataset slug:        {slug}")
    print(f"Output base dir:     {base_dir}/")
    print(f"Experiments to run:  {args.exp}")

    if 1 in args.exp:
        run_exp1(
            args.json_path,
            output_dir=f"{base_dir}/diff_histograms",
            nrows=args.nrows,
            ncols=args.ncols,
        )

    if 2 in args.exp:
        run_exp2(
            args.json_path,
            output_dir=f"{base_dir}/boxplots",
            nrows=args.nrows,
            ncols=args.ncols,
        )

    if 3 in args.exp:
        run_exp3(
            args.json_path,
            output_dir=f"{base_dir}/polar",
            nrows=args.nrows,
            ncols=args.ncols,
        )

    if 4 in args.exp:
        run_exp4(
            args.json_path,
            output_dir=f"{base_dir}/heatmap",
            threshold=args.threshold,
        )

    print(f"\nAll selected experiments complete. Outputs in {base_dir}/")


if __name__ == "__main__":
    main()
