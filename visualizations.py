# visualizations.py

import torch
from collections import defaultdict
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_json

def get_biword_scores(path: str = "output/multi_word_output.json"):
    """
    Build aggregated attention scores for *bi-token* words from the AI-friendly JSON.

    Input JSON shape (per word, see analysis.aggregate_multi_token_word_attentions_ai):
      {
        "<word>": {
          "occurrences": [
            {
              "token_indices": [i, j],
              "attentions": {
                "layers": [
                  { "heads": [head_0_tensor, head_1_tensor, ...] },
                  ...
                ]
              }
            },
            ...
          ]
        },
        ...
      }
    Returns:
      Nested dict: result[layer_idx][head_idx] = {
            "first":  [float, ...],  # attention values for subtoken position 0
            "second": [float, ...],  # attention values for subtoken position 1
      }
    """
    in_map = load_json(path)

    mw_map = in_map['main_data']
    result = defaultdict(lambda: defaultdict(lambda: {"first": [], "second": []}))
    for word in mw_map:
        for occurrence in mw_map[word]["occurrences"]:
            layers = occurrence["attentions"]["layers"]
            for layer_idx, layer in enumerate(layers):
                for head_idx, scores in enumerate(layer["heads"]):
                    # scores shape: [num_subtokens, num_valid_rows]
                    # Convert tensor-like objects to lists if needed (json already does it)
                    if len(scores) == 2:
                        first_vals = scores[0]
                        second_vals = scores[1]
                        result[layer_idx][head_idx]["first"].extend(first_vals)
                        result[layer_idx][head_idx]["second"].extend(second_vals)
    return result

def plot_layer_histogram_max_per_head(
    path: str = "output/multi_word_output.json",
    output_dir: str = "visuals/all/histograms/five_paragraphs",
    nrows: int = 8,
    ncols: int = 4,
):
    """
    For every layer, plots a histogram across heads, where each head's value is:
    - 0 if max in "first" > max in "second"
    - 1 if max in "second" >= max in "first"
    Follows a subplot layout like plot_per_layer_box_whisker.
    """
    import os
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    layer_head_data = get_biword_scores(path)
    num_layers = len(layer_head_data)

    # Validation: nrows * ncols must equal num_layers (we use same arrangement as boxplot)
    if nrows * ncols != len(next(iter(layer_head_data.values()))):
        raise ValueError(
            f"nrows * ncols ({nrows} * {ncols} = {nrows*ncols}) does not match the number of heads per layer ({len(next(iter(layer_head_data.values())))})."
        )

    for layer in range(num_layers):
        layer_data = layer_head_data[layer]
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)

        for head in range(nrows * ncols):
            row = head // ncols
            col = head % ncols
            ax = axes[row, col]

            first = layer_data[head]["first"]
            second = layer_data[head]["second"]

            # Determine for each occurrence if max(first) > max(second): 0 else 1
            choices = []
            if first or second:
                # If either list is non-empty
                f_max = max(first) if first else float('-inf')
                s_max = max(second) if second else float('-inf')
                if f_max > s_max:
                    choices.append(0)
                else:
                    choices.append(1)
            else:
                # No data
                choices.append(-1)

            # For visualization: just plot histogram with 0 or 1 for this head (can be only 1 data point)
            ax.hist(choices, bins=[-0.5, 0.5, 1.5], rwidth=0.8, color="#4C72B0")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["max[0] > max[1]", "max[1] >= max[0]"])
            ax.set_xlabel("Max position")
            ax.set_ylabel("Count")
            ax.set_title(f"Head {head}")

        fig.suptitle(f"Layer {layer} — Max attention subtoken position per head", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"layer_{layer}_bitoken_max_histogram.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_per_layer_box_whisker(
    json_path="output/multi_word_output.json",
    output_dir="visuals/three_paragraphs",
    nrows=8,
    ncols=4
):
    os.makedirs(output_dir, exist_ok=True)

    layer_head_data = get_biword_scores(json_path)
    num_layers = len(layer_head_data)
    colors = ["#4C72B0", "#DD8452"]

    # Validation: nrows * ncols must equal num_layers
    if nrows * ncols != num_layers:
        raise ValueError(
            f"nrows * ncols ({nrows} * {ncols} = {nrows*ncols}) does not match the number of layers ({num_layers})"
        )

    for layer in range(num_layers):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)

        for head in range(nrows * ncols):
            row = head // ncols
            col = head % ncols
            ax = axes[row, col]

            first = layer_head_data[layer][head]["first"]
            second = layer_head_data[layer][head]["second"]

            bp = ax.boxplot(
                [first, second] if first and second else [first or [0], second or [0]],
                patch_artist=True,
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Compute max values for annotation
            max_first = max(first) if first else 0
            max_second = max(second) if second else 0

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["0", "1"])
            ax.set_xlabel("Subtoken position")
            ax.set_ylabel("Attention")
            # Two-line title: Head i\nmax[0]: val, max[1]: val
            ax.set_title(
                f"Head {head}\nmax[0]: {max_first:.4f}, max[1]: {max_second:.4f}"
            )
            ax.set_ylim(0, 1)  # Set the max y value to 1 for all plots

        fig.suptitle(f"Layer {layer} — Bitoken word attention by subtoken position", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"layer_{layer}_bitoken_boxplot.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print(f"Done — {num_layers} layer plots saved to {output_dir}/")


def combine_layer_plots(
    input_dir,
    ncols=1,
    nrows=32,
    output_path = None,
    type="boxplot"  # "boxplot" or "histogram"
):
    """
    Stitches all per-layer PNGs from input_dir into one large grid image.
    'type' determines which filename template to use:
      - 'boxplot': 'layer_{i}_bitoken_boxplot.png'
      - 'histogram': 'layer_{i}_max_histogram.png'
    """
    from PIL import Image
    import os

    if type == "boxplot":
        file_template = "layer_{i}_bitoken_boxplot.png"
    elif type == "histogram":
        file_template = "layer_{i}_max_histogram.png"
    elif type == "diff_histogram":
        file_template = "layer_{i}_diff_histogram.png"
    elif type == "polar_grid":
        file_template = "layer_{i}_polar_grid.png"
    else:
        raise ValueError(f"Unknown plot type: {type}")

    files = [file_template.format(i=i) for i in range(nrows * ncols)]
    imgs = [Image.open(os.path.join(input_dir, f)) for f in files]
    w, h = imgs[0].size

    combined = Image.new("RGB", (w * ncols, h * nrows), "white")
    for i, img in enumerate(imgs):
        combined.paste(img, ((i % ncols) * w, (i // ncols) * h))

    if output_path is None:
        output_path = os.path.join(input_dir, f"all_layers_combined_{type}.png")
    combined.save(output_path, dpi=(150, 150))
    print(f"Saved {output_path} — {combined.size[0]}x{combined.size[1]} pixels")


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def plot_hypothesis_rate_heatmap(
    rates: dict,
    output_dir: str = "visuals/experiments/heatmap",
) -> None:
    """
    Plots a 2D heatmap of hypothesis rates (last-subtoken dominance) indexed
    by (layer, head). Each cell shows the fraction of occurrences where
    mean(tok_1) > mean(tok_0) for that (layer, head) pair.

    Args:
        rates:      output of analysis.compute_head_hypothesis_rates
                    { (layer_idx, head_idx): float in [0, 1] }
        output_dir: directory to save the PNG into
    """
    os.makedirs(output_dir, exist_ok=True)

    if not rates:
        print("[plot_hypothesis_rate_heatmap] No data to plot.")
        return

    num_layers = max(layer for layer, _ in rates) + 1
    num_heads = max(head for _, head in rates) + 1

    grid = np.zeros((num_layers, num_heads))
    for (layer, head), rate in rates.items():
        grid[layer, head] = rate

    fig_w = max(10, num_heads * 0.55)
    fig_h = max(6, num_layers * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(grid, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
    plt.colorbar(im, ax=ax, label="Hypothesis rate (tok_1 > tok_0)")

    font_size = max(4, min(7, int(80 / max(num_layers, num_heads))))
    for layer in range(num_layers):
        for head in range(num_heads):
            ax.text(
                head, layer, f"{grid[layer, head]:.2f}",
                ha="center", va="center", fontsize=font_size,
                color="black" if 0.2 < grid[layer, head] < 0.8 else "white",
            )

    ax.set_xlabel("Head", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(range(num_heads), fontsize=7)
    ax.set_yticklabels(range(num_layers), fontsize=7)
    ax.set_title("Hypothesis rate — last subtoken dominance (tok_1 > tok_0)", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "hypothesis_rate_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_layer_hypothesis_bar(
    layer_rates: dict,
    output_dir: str = "visuals/experiments/heatmap",
    threshold: float = 0.5,
) -> None:
    """
    Plots a horizontal bar chart of mean hypothesis rates per layer.
    Each bar shows the average fraction of heads (across all heads in that layer)
    where tok_1 > tok_0. A vertical reference line marks the threshold.

    Args:
        layer_rates: output of analysis.compute_layer_hypothesis_rates
                     { layer_idx: float in [0, 1] }
        output_dir:  directory to save the PNG into
        threshold:   reference line position (default 0.5)
    """
    os.makedirs(output_dir, exist_ok=True)

    if not layer_rates:
        print("[plot_layer_hypothesis_bar] No data to plot.")
        return

    layers = sorted(layer_rates)
    rates = [layer_rates[l] for l in layers]
    colors = ["#4C72B0" if r >= threshold else "#DD8452" for r in rates]

    fig, ax = plt.subplots(figsize=(6, max(4, len(layers) * 0.35)))
    bars = ax.barh(layers, rates, color=colors, edgecolor="none", height=0.7)
    ax.axvline(threshold, color="black", lw=1, linestyle="--", label=f"threshold={threshold}")

    for bar, rate in zip(bars, rates):
        ax.text(
            min(rate + 0.01, 0.98), bar.get_y() + bar.get_height() / 2,
            f"{rate:.3f}", va="center", fontsize=7,
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean hypothesis rate (tok_1 > tok_0)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_yticks(layers)
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title("Mean hypothesis rate per layer", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "layer_hypothesis_rate_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_diff_heatmap(
    diffs: dict,
    output_dir: str = "visuals/experiments/exp2_differences",
) -> None:
    """
    Plots a 2D heatmap of mean attention difference per occurrence (tok_1 − tok_0)
    indexed by (layer, head). Each cell shows the average diff across occurrences
    for that (layer, head) pair; positive = last subtoken heavier on average,
    negative = first subtoken heavier.

    Args:
        diffs:      output of analysis.get_biword_score_pairs_diff
                    { (layer_idx, head_idx): [diff, ...] }
        output_dir: directory to save the PNG into
    """
    os.makedirs(output_dir, exist_ok=True)

    if not diffs:
        print("[plot_diff_heatmap] No data to plot.")
        return

    num_layers = max(layer for layer, _ in diffs) + 1
    num_heads = max(head for _, head in diffs) + 1

    grid = np.zeros((num_layers, num_heads))
    for (layer, head), values in diffs.items():
        grid[layer, head] = float(np.mean(values)) if values else 0.0

    abs_max = np.abs(grid).max() or 1.0

    fig_w = max(10, num_heads * 0.55)
    fig_h = max(6, num_layers * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(grid, vmin=-abs_max, vmax=abs_max, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, ax=ax, label="Mean diff (tok_1 − tok_0)")

    font_size = max(4, min(7, int(80 / max(num_layers, num_heads))))
    for layer in range(num_layers):
        for head in range(num_heads):
            ax.text(
                head, layer, f"{grid[layer, head]:.3f}",
                ha="center", va="center", fontsize=font_size,
                color="black" if abs(grid[layer, head]) < abs_max * 0.7 else "white",
            )

    ax.set_xlabel("Head", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(range(num_heads), fontsize=7)
    ax.set_yticklabels(range(num_layers), fontsize=7)
    ax.set_title("Mean pairwise attention difference (tok_1 − tok_0)", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "diff_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_diff_contrast_heatmap(
    contrasts: dict,
    output_dir: str = "visuals/experiments/exp_contrast",
) -> None:
    """
    Plots a 2D heatmap of mean Michelson contrast per occurrence —
    (mean(tok_1) − mean(tok_0)) / (mean(tok_1) + mean(tok_0)) —
    indexed by (layer, head). Each cell shows the average contrast across
    occurrences for that (layer, head) pair; positive (red) = last subtoken
    dominates on average, negative (blue) = first subtoken dominates.

    Unlike the raw-diff heatmap, this metric is scale-invariant: the same
    absolute difference carries more weight when the underlying values are small
    (Michelson contrast formula).

    Args:
        contrasts:  output of analysis.get_biword_score_pairs_contrast
                    { (layer_idx, head_idx): [contrast, ...] }
        output_dir: directory to save the PNG into
    """
    os.makedirs(output_dir, exist_ok=True)

    if not contrasts:
        print("[plot_diff_contrast_heatmap] No data to plot.")
        return

    num_layers = max(layer for layer, _ in contrasts) + 1
    num_heads = max(head for _, head in contrasts) + 1

    grid = np.zeros((num_layers, num_heads))
    for (layer, head), values in contrasts.items():
        grid[layer, head] = float(np.mean(values)) if values else 0.0

    abs_max = np.abs(grid).max() or 1.0

    fig_w = max(10, num_heads * 0.55)
    fig_h = max(6, num_layers * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(grid, vmin=-abs_max, vmax=abs_max, aspect="auto", cmap="coolwarm_r")
    plt.colorbar(im, ax=ax, label="Mean Michelson contrast  (tok_1 − tok_0) / (tok_1 + tok_0)")

    font_size = max(4, min(7, int(80 / max(num_layers, num_heads))))
    for layer in range(num_layers):
        for head in range(num_heads):
            ax.text(
                head, layer, f"{grid[layer, head]:.3f}",
                ha="center", va="center", fontsize=font_size,
                color="black" if abs(grid[layer, head]) < abs_max * 0.7 else "white",
            )

    ax.set_xlabel("Head", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_xticks(range(num_heads))
    ax.set_yticks(range(num_layers))
    ax.set_xticklabels(range(num_heads), fontsize=7)
    ax.set_yticklabels(range(num_layers), fontsize=7)
    ax.set_title("Mean Michelson contrast per head  —  (tok_1 − tok_0) / (tok_1 + tok_0)", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "diff_contrast_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def plot_layer_contrast_bar(
    layer_contrasts: dict,
    output_dir: str = "visuals/experiments/exp_contrast",
) -> None:
    """
    Plots a horizontal bar chart of mean Michelson contrast per layer.
    Each bar shows the average contrast across all heads in that layer.
    A vertical reference line at 0 marks the boundary between tok_0 dominance
    (negative/red) and tok_1 dominance (positive/blue).

    Args:
        layer_contrasts: output of analysis.compute_layer_contrast_means
                         { layer_idx: float in [-1, 1] }
        output_dir:      directory to save the PNG into
    """
    os.makedirs(output_dir, exist_ok=True)

    if not layer_contrasts:
        print("[plot_layer_contrast_bar] No data to plot.")
        return

    layers = sorted(layer_contrasts)
    contrasts = [layer_contrasts[l] for l in layers]
    colors = ["#4C72B0" if c >= 0 else "#DD8452" for c in contrasts]

    fig, ax = plt.subplots(figsize=(6, max(4, len(layers) * 0.35)))
    bars = ax.barh(layers, contrasts, color=colors, edgecolor="none", height=0.7)
    ax.axvline(0, color="black", lw=1, linestyle="--", label="0 (no preference)")

    for bar, c in zip(bars, contrasts):
        x = c + 0.01 if c >= 0 else c - 0.01
        ha = "left" if c >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{c:.3f}", va="center", ha=ha, fontsize=7)

    ax.set_xlim(-1, 1)
    ax.set_xlabel("Mean Michelson contrast  (tok_1 − tok_0) / (tok_1 + tok_0)", fontsize=10)
    ax.set_ylabel("Layer", fontsize=10)
    ax.set_yticks(layers)
    ax.set_yticklabels([f"Layer {l}" for l in layers], fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.set_title("Mean Michelson contrast per layer", fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "layer_contrast_bar.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    output_path = "visuals/all/boxplots/five_paragraphs"
    # plot_layer_histogram_max_per_head()
    plot_per_layer_box_whisker(output_dir = output_path)
    combine_layer_plots(output_path)
    # combine_layer_plots(input_dir = "visuals/all/histograms/five_paragraphs", type = "histogram")

