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


if __name__ == "__main__":
    output_path = "visuals/all/boxplots/five_paragraphs"
    # plot_layer_histogram_max_per_head()
    plot_per_layer_box_whisker(output_dir = output_path)
    combine_layer_plots(output_path)
    # combine_layer_plots(input_dir = "visuals/all/histograms/five_paragraphs", type = "histogram")

