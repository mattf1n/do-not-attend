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


def get_biword_scores(path: str = "output/multi_word_output_ai.json"):
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
    mw_map = load_json(path)
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


def plot_per_layer_box_whisker(
    json_path="output/multi_word_output_ai.json",
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

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["0", "1"])
            ax.set_xlabel("Subtoken position")
            ax.set_ylabel("Attention")
            ax.set_title(f"Head {head}")

        fig.suptitle(f"Layer {layer} — Bitoken word attention by subtoken position", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"layer_{layer}_bitoken_boxplot.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print(f"Done — {num_layers} layer plots saved to {output_dir}/")


def combine_layer_plots(input_dir, output_path=None, ncols=4, nrows=8):
    """
    Stitches all 32 per-layer boxplot PNGs from input_dir into one large grid image.
    """
    from PIL import Image
    import os

    files = [f"layer_{i}_bitoken_boxplot.png" for i in range(nrows * ncols)]
    imgs = [Image.open(os.path.join(input_dir, f)) for f in files]
    w, h = imgs[0].size

    combined = Image.new("RGB", (w * ncols, h * nrows), "white")
    for i, img in enumerate(imgs):
        combined.paste(img, ((i % ncols) * w, (i // ncols) * h))

    if output_path is None:
        output_path = os.path.join(input_dir, "all_layers_combined.png")
    combined.save(output_path, dpi=(150, 150))
    print(f"Saved {output_path} — {combined.size[0]}x{combined.size[1]} pixels")


if __name__ == "__main__":
plot_per_layer_box_whisker()
combine_layer_plots("visuals/three_paragraphs")

