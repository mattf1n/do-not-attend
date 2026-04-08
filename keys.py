import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

from model import get_model

matplotlib.use("Agg")


def get_polar_coordinates(a, b):
    """
    Returns polar coordinates of vector a relative to vector b as reference.
        r     = ||a|| / ||b||
        theta = angle between a and b (radians)
    """
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    cos_theta = torch.dot(a, b) / (na * nb)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.arccos(cos_theta)
    r = na / nb
    return r, theta


def plot_polar_point(r, theta, save_path="plot.png"):
    """
    Plot a single polar coordinate (r, theta) where theta is in radians.
    """
    r, theta = float(r), float(theta)
    theta_deg = np.degrees(theta)

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection="polar"), facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetalim(0, np.pi)

    r_max = max(r * 1.35, 1.0)
    ax.set_ylim(0, r_max)
    ax.set_thetagrids([0, 45, 90, 135, 180])
    ax.set_rlabel_position(90)

    ax.plot([0, theta], [0, r], color="navy", lw=2.5, zorder=3)
    ax.scatter([theta], [r], color="navy", s=25, zorder=4)

    r_arc = r * 0.25
    theta_arc = np.linspace(0, theta, 100)
    ax.plot(theta_arc, np.full_like(theta_arc, r_arc), color="black", lw=1.2)

    ax.text(
        theta, r * 0.5, f"({r:.2f}, {theta_deg:.2f}°)",
        ha="center", va="bottom", fontsize=11, color="navy", fontweight="bold",
        rotation=theta_deg, rotation_mode="anchor"
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def extract_key_vectors(
    text: str,
    model,
    tokenizer,
    multi_token_words_map: dict,
) -> dict:
    """
    Runs inference with use_cache=True to get past_key_values, then extracts
    key vectors for every bi-token word occurrence per (layer, head).

    Args:
        text:                  raw input string
        model:                 loaded HuggingFace causal LM
        tokenizer:             corresponding tokenizer
        multi_token_words_map: output of tokenization.get_multi_token_words

    Returns:
        { (layer_idx, head_idx): {"k0": [tensor, ...], "k1": [tensor, ...]} }
        Each list contains one head_dim-length tensor per bi-token occurrence.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)

    pkv = outputs.past_key_values
    num_layers = len(pkv)

    accum = defaultdict(lambda: {"k0": [], "k1": []})

    for word, info in multi_token_words_map.items():
        for occ in info["occurrences"]:
            token_indices = occ["token_indices"]
            if len(token_indices) != 2:
                continue
            for layer_idx in range(num_layers):
                # shape: (batch, num_kv_heads, seq_len, head_dim)
                layer_keys = pkv.layers[layer_idx].keys
                num_heads = layer_keys.shape[1]
                for head_idx in range(num_heads):
                    k0 = layer_keys[0, head_idx, token_indices[0], :].float().cpu()
                    k1 = layer_keys[0, head_idx, token_indices[1], :].float().cpu()
                    accum[(layer_idx, head_idx)]["k0"].append(k0)
                    accum[(layer_idx, head_idx)]["k1"].append(k1)

    # Average all collected k0 and k1 vectors per (layer, head).
    return {
        key: {
            "k0": torch.stack(val["k0"]).mean(dim=0),
            "k1": torch.stack(val["k1"]).mean(dim=0),
        }
        for key, val in accum.items()
    }


def compute_polar_per_head(
    key_vectors: dict,
) -> dict:
    """
    For every (layer, head), computes polar coordinates (r, theta) between the
    mean k0 and mean k1 key vectors. k1 is treated as the reference vector.

    Args:
        key_vectors: output of extract_key_vectors
                     { (layer_idx, head_idx): {"k0": tensor, "k1": tensor} }
                     where k0 and k1 are mean key vectors of shape (head_dim,).

    Returns:
        { (layer_idx, head_idx): (r, theta) }
        r     = ||mean_k0|| / ||mean_k1||
        theta = angle between mean_k0 and mean_k1 in radians
    """
    return {
        (layer_idx, head_idx): get_polar_coordinates(vecs["k0"], vecs["k1"])
        for (layer_idx, head_idx), vecs in key_vectors.items()
    }


def plot_polar_grid(
    polar_data: dict,
    output_dir: str = "visuals/experiments/exp3_polar",
    nrows: int = 8,
    ncols: int = 4,
) -> None:
    """
    For each layer, plots a grid of polar plots — one subplot per head —
    showing the (r, theta) of the mean k0 vs mean k1 key vectors.

    Args:
        polar_data: output of compute_polar_per_head
                    { (layer_idx, head_idx): (r, theta) }
        output_dir: directory to save PNGs into
        nrows:      subplot rows
        ncols:      subplot columns
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if not polar_data:
        print("[plot_polar_grid] No data to plot.")
        return

    num_layers = max(layer for layer, _ in polar_data) + 1
    num_heads = nrows * ncols

    for layer in range(num_layers):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 3, nrows * 3),
            subplot_kw=dict(projection="polar"),
            squeeze=False,
        )

        for head in range(num_heads):
            row = head // ncols
            col = head % ncols
            ax = axes[row, col]

            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_thetalim(0, np.pi)
            ax.set_thetagrids([0, 45, 90, 135, 180])

            if (layer, head) in polar_data:
                r, theta = polar_data[(layer, head)]
                r, theta = float(r), float(theta)
                ax.plot([0, theta], [0, r], color="navy", lw=2)
                ax.scatter([theta], [r], s=30, color="navy", zorder=5,
                           label=f"({r:.2f}, {np.degrees(theta):.1f}°)")
                ax.legend(fontsize=6, loc="upper right")
            else:
                ax.text(np.pi / 2, 0.5, "no data", ha="center", va="center", fontsize=8,
                        transform=ax.transAxes)

            ax.set_title(f"H{head}", fontsize=8, pad=2)
            ax.tick_params(labelsize=6)

        fig.suptitle(f"Layer {layer} — Key vector polar coords (k0 vs k1)", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"layer_{layer}_polar_grid.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print(f"Done — {num_layers} layer plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys
    from utils import load_json
    from tokenization import get_multi_token_words

    json_path = sys.argv[1] if len(sys.argv) > 1 else "output/multi_word_output.json"
    in_map = load_json(json_path)
    text = in_map["text"]
    multi_token_words_map = in_map["main_data"]

    model, tokenizer = get_model()
    key_vectors = extract_key_vectors(text, model, tokenizer, multi_token_words_map)
    del model

    polar_data = compute_polar_per_head(key_vectors)
    plot_polar_grid(polar_data)
