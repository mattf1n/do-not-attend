"""
run_experiments.py

Entry point for running all hypothesis experiments against a saved output JSON
(produced by main.py). Does not re-run the model except for exp3 (key vectors).

Usage:
    python run_experiments.py                                    # uses default path
    python run_experiments.py output/my_output.json              # single JSON
    python run_experiments.py output/my_output.json --exp 2 3    # run only exp 2 and 3
    python run_experiments.py --folder output/500_tokens/        # all JSONs in folder
    python run_experiments.py --folder output/500_tokens/ --exp 4 6  # subset of exps

Experiments:
    2 — Box-whisker plots: attention score distributions per subtoken per head
    3 — Key vector polar coordinates: geometric comparison of subtoken key vectors
        (requires re-running the model to extract past_key_values)
    4 — Hypothesis rate heatmap: 2D grid of (layer x head) hypothesis rates
    5 — Per-layer hypothesis rate bar chart: mean rate across heads per layer
    6 — Michelson contrast heatmap: scale-invariant (tok_1 - tok_0) / (tok_1 + tok_0) per head
    7 — Per-layer Michelson contrast bar chart: mean contrast across heads per layer

Output folder structure:
    visuals/<dataset_slug>/
        boxplots/
        polar/
        rate_head_heatmap/
        rate_layer_bar/
        contrast_heatmap/
        contrast_layer_bar/

    The dataset_slug is derived from the JSON filename, e.g.:
        output/10_paragraphs_2-token_max_word_output.json
        → visuals/10_paragraphs_2-token_max_word/

    In --folder mode each JSON is nested under the input folder name, e.g.:
        output/500_tokens/PubMed_Abstracts_500tokens.json
        → visuals/500_tokens/PubMed_Abstracts_500tokens/
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


def get_dataset_slug(json_path: str, npz_path: str = None) -> str:
    """Derives a clean folder name from the JSON filename or npz folder."""
    if npz_path:
        return Path(npz_path).name
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


def run_exp5(
    json_path: str,
    output_dir: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    """
    Experiment 5: Per-layer hypothesis rate bar chart.
    Averages the hypothesis rate across all heads for each layer and plots
    a horizontal bar chart, making it easy to see which layers most consistently
    attend more to the last subtoken.
    """
    print("\n=== Experiment 5: Per-Layer Hypothesis Rate Bar Chart ===")
    rates = compute_layer_hypothesis_rates(json_path)
    passing = sum(1 for v in rates.values() if v >= threshold)
    print(f"Threshold: {threshold:.0%} | "
          f"{passing}/{len(rates)} layers pass ({100*passing/len(rates):.1f}%)")
    plot_layer_hypothesis_bar(rates, output_dir=output_dir, threshold=threshold)


def run_exp6(
    json_path: str,
    output_dir: str,
) -> None:
    """
    Experiment 6: Michelson contrast heatmap.
    Plots a 2D (layer x head) heatmap of mean (tok_1 - tok_0) / (tok_1 + tok_0),
    a scale-invariant measure of last-subtoken dominance.
    """
    print("\n=== Experiment 6: Michelson Contrast Heatmap ===")
    contrasts = get_biword_score_pairs_contrast(json_path)
    print(f"Loaded {sum(len(v) for v in contrasts.values())} contrast values "
          f"across {len(contrasts)} (layer, head) pairs.")
    plot_diff_contrast_heatmap(contrasts, output_dir=output_dir)


def run_exp7(
    json_path: str,
    output_dir: str,
) -> None:
    """
    Experiment 7: Per-layer Michelson contrast bar chart.
    Averages the contrast across all heads for each layer and plots a horizontal
    bar chart, making it easy to see which layers most consistently attend more
    to the last subtoken on a scale-invariant basis.
    """
    print("\n=== Experiment 7: Per-Layer Michelson Contrast Bar Chart ===")
    layer_contrasts = compute_layer_contrast_means(json_path)
    plot_layer_contrast_bar(layer_contrasts, output_dir=output_dir)


def _run_all_exps(json_path: str, base_dir: str, args) -> None:
    """Run all selected experiments for a single JSON file, writing into base_dir."""
    if 2 in args.exp:
        run_exp2(
            json_path,
            output_dir=f"{base_dir}/boxplots",
            nrows=args.nrows,
            ncols=args.ncols,
        )

    if 3 in args.exp:
        run_exp3(
            json_path,
            output_dir=f"{base_dir}/polar",
            nrows=args.nrows,
            ncols=args.ncols,
        )

    if 4 in args.exp:
        run_exp4(
            json_path,
            output_dir=f"{base_dir}/rate_head_heatmap",
            threshold=args.threshold,
        )

    if 5 in args.exp:
        run_exp5(
            json_path,
            output_dir=f"{base_dir}/rate_layer_bar",
            threshold=args.threshold,
        )

    if 6 in args.exp:
        run_exp6(
            json_path,
            output_dir=f"{base_dir}/contrast_heatmap",
        )

    if 7 in args.exp:
        run_exp7(
            json_path,
            output_dir=f"{base_dir}/contrast_layer_bar",
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
        help="Which experiments to run (default: all). E.g. --exp 2 4",
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

    if args.folder:
        json_files = sorted(Path(args.folder).glob("*.json"))
        if not json_files:
            print(f"No .json files found in {args.folder}")
            return
        print(f"Folder mode: found {len(json_files)} JSON file(s) in {args.folder}")
        print(f"Experiments to run: {args.exp}\n")
        folder_name = Path(args.folder).name
        for json_file in json_files:
            json_path = str(json_file)
            slug = get_dataset_slug(json_path)
            base_dir = get_unique_base_dir(f"{folder_name}/{slug}")
            print(f"--- Input: {json_path}")
            print(f"    Output base dir: {base_dir}/")
            _run_all_exps(json_path, base_dir, args)
    else:
        is_temp = False
        try:
            json_path, is_temp = resolve_json_path(args)
            slug = get_dataset_slug(json_path, npz_path=args.npz)
            base_dir = get_unique_base_dir(slug)

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
