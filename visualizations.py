# visualizations.py
from utils import test_pipeline, get_model, get_multi_token_words
from visualizations import plot_2token_keys_per_head, plot_by_word_length

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



def plot_box_whisper_all_bi_token_words(json_path="test_pipeline_results_all_paragraph.json"):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    plt.close("all")

    zeros, ones = get_biword_split(json_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([zeros, ones], patch_artist=True, tick_labels=["0", "1"])
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Position of subtoken")
    ax.set_ylabel("Attention score")
    ax.set_title(f"2-token words attention by subtoken position")

    plt.tight_layout()
    plt.savefig("boxplot_2token_words.png", dpi=150)
    plt.close()


def get_biword_split(json_path="test_pipeline_results_all_paragraph.json"):
    with open(json_path) as f:
        out = json.load(f)

    scores = out["scores"]

    words = {word: t for word, t in scores.items() if len(t) == 2}

    zeros, ones = [], []

    for word, subtokens in words.items():
        for i, subtoken_data in enumerate(subtokens.values()):
            for layer, heads in subtoken_data.items():
                for head, values in heads.items():
                    if i == 0:
                        zeros += values
                    else:
                        ones += values

    return (zeros, ones)
