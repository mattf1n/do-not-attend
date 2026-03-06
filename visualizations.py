# visualizations.py
from utils import test_pipeline, get_model, get_multi_token_words
import torch
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_by_word_length_hist_summary(json_path="sample_results.json"):
    import json
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    plt.close("all")

    with open(json_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, length in zip(axes, [2, 3, 4]):
        words = {w: t for w, t in data.items() if len(t) == length}
        argmax_positions = []
        for word, tokens in words.items():
            vals = list(tokens.values())
            argmax_positions.append(vals.index(max(vals)))

        sns.histplot(argmax_positions, discrete=True, ax=ax)
        ax.set_xlabel("Position of highest attention subtoken")
        ax.set_ylabel("Count")
        ax.set_title(f"{length}-token words (n={len(words)})")
        ax.set_xticks(range(length))

    plt.tight_layout()
    plt.savefig("argmax_by_word_length.png", dpi=150)


def get_biword_split_old(json_path="test_pipeline_results_all_paragraph.json"):
    import json
    with open(json_path) as f:
        out = json.load(f)

    scores = out["scores"]
    words = {word: t for word, t in scores.items() if len(t) == 2} # only get words with 2 subtokens
    # words_more_than_2 = [w for w, t in scores.items() if len(t) > 2]
    # print(f"Words with more than 2 subtokens: {words_more_than_2}")
    # only 

    zeros, ones = [],[]
    for word, subtokens in words.items():
        for i, subtoken_data in enumerate(subtokens.values()):
            for layer, heads in subtoken_data.items():
                for head, values in heads.items():
                    if i == 0:
                        zeros += values
                    else:
                        ones += values
    return (zeros, ones)


def get_biword_scores_by_layer_head(json_path="test_pipeline_results_all_paragraph.json"):
    """
    Returns a nested dict: {layer: {head: {"first": [...], "second": [...]}}}
    aggregated across all bitoken words.
    """
    with open(json_path) as f:
        out = json.load(f)

    scores = out["scores"]
    words = {word: t for word, t in scores.items() if len(t) == 2}

    result = defaultdict(lambda: defaultdict(lambda: {"first": [], "second": []}))

    for word, subtokens in words.items():
        for i, (subtoken_name, subtoken_data) in enumerate(subtokens.items()):
            for layer, heads in subtoken_data.items():
                for head, values in heads.items():
                    if i == 0:
                        result[int(layer)][int(head)]["first"].extend(values)
                    else:
                        result[int(layer)][int(head)]["second"].extend(values)

    return result


def plot_per_layer_box_whisker(
    json_path="test_pipeline_results_all_paragraph.json",
    output_dir="visuals",
    nrows=8,
    ncols=4
):
    import os
    os.makedirs(output_dir, exist_ok=True)

    layer_head_data = get_biword_scores_by_layer_head(json_path)
    num_layers = len(layer_head_data)
    colors = ["#4C72B0", "#DD8452"]

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
    # plot_per_layer_box_whisker(json_path="test_pipeline_results_all_paragraph.json")
    # combine_layer_plots("visuals/one_paragraph")


    