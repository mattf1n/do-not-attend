"""
qkv_vectors.py — Q/K/V vector extraction and analysis for multi-token words.

Uses TransformerLens ActivationCache (post-RoPE Q/K, V from hook_out), not HF
past_key_values.

Main workflow:
    1. save_slot_vectors  — from a TL cache, write one .pt per slot (q0,q1,k0,k1,v0,v1)
    2. collect_vectors    — merge two slot files into the pair schema for analysis
    3. micro/macro_average_vectors — reduce to one (head_dim,) vector per (layer, head)
    4. compute_polar_per_head — polar coords between the two averaged slots

Slot file schema (one list per (layer, head)):
    { word: { (layer_idx, head_idx): [occ1, occ2, ...] } }

Paired schema (after collect_vectors):
    { word: { (layer_idx, head_idx): { name_a: [occ...], name_b: [occ...] } } }
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import WORD_CATEGORIES, classify_word

matplotlib.use("Agg")

# role + subtoken index (0 = first bitoken piece, 1 = second)
SLOTS = ("q0", "q1", "k0", "k1", "v0", "v1")
_ROLE_RANK = {"q": 0, "k": 1, "v": 2}  # tie-break when both subtoken indices match

# TL cache key suffix per role (post-RoPE for q/k)
_ROLE_HOOK = {
    "q": "hook_rot_q",
    "k": "hook_rot_k",
    "v": "v.hook_out",
}


def _parse_slot(slot: str) -> tuple[str, int]:
    """Parse 'q0' → ('q', 0)."""
    if not re.fullmatch(r"[qkv][01]", slot):
        raise ValueError(f"Unknown slot {slot!r}. Valid: {list(SLOTS)}")
    return slot[0], int(slot[1])


def order_slot_pair(slot_x: str, slot_y: str) -> tuple[str, str]:
    """
    Order two slots into (name_a, name_b) for polar analysis.

    Pair-order rule (subtoken index is the 0/1 in the slot name):
      - A = earlier subtoken, B = later subtoken
      - Allowed index pairs only: (0,0), (0,1), (1,1) — never leave (1,0) as-is;
        if the raw pair is (1,0), swap so A has 0 and B has 1
      - If both indices match (00 or 11): tie-break by role order q < k < v

    Polar radius uses B as reference:
      r = ||name_a|| / ||name_b||
      theta = angle between name_a and name_b

    Examples:
      order_slot_pair("q1", "k0") → ("k0", "q1")   # swap to (0,1)
      order_slot_pair("q0", "k0") → ("q0", "k0")   # both 0; q < k
      order_slot_pair("k0", "k1") → ("k0", "k1")   # (0,1)
    """
    role_x, idx_x = _parse_slot(slot_x)
    role_y, idx_y = _parse_slot(slot_y)
    if slot_x == slot_y:
        raise ValueError(f"Need two distinct slots, got {slot_x!r} twice")

    # Prefer earlier subtoken as A, later as B
    if idx_x < idx_y:
        return slot_x, slot_y
    if idx_y < idx_x:
        return slot_y, slot_x
    # Same subtoken index (00 or 11): earlier role wins as A
    if _ROLE_RANK[role_x] <= _ROLE_RANK[role_y]:
        return slot_x, slot_y
    return slot_y, slot_x


def iter_ordered_slot_pairs() -> list[tuple[str, str]]:
    """
    All C(6,2)=15 unordered pairs from SLOTS, each ordered by order_slot_pair.

    Every returned (name_a, name_b) satisfies the pair-order rule so that
    r = ||name_a|| / ||name_b|| always means earlier/equal-tie A over later B.
    """
    return [order_slot_pair(x, y) for x, y in combinations(SLOTS, 2)]


def _pair_keys(sample_inner: dict) -> tuple[str, str]:
    """Return the two slot names in an inner {(layer,head): {a:..., b:...}} value."""
    keys = list(sample_inner.keys())
    if len(keys) != 2:
        raise ValueError(f"Expected exactly two slot keys, got {keys!r}")
    return keys[0], keys[1]


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
# EXTRACTION / I/O
# ─────────────────────────────────────────────────────────────────────────────


def save_slot_vectors(
    cache,
    multi_token_words_map: dict,
    out_dir: str,
    slots: tuple[str, ...] = SLOTS,
    prefix: str = "",
    n_layers: int | None = None,
) -> dict[str, str]:
    """
    From a TL ActivationCache, extract multi-token vectors and write one .pt per slot.

    Each file:
      { word: { (layer, head): [occ0, occ1, ...] } }

    Occurrence order is identical across slots so files stay alignable for mixing
    (e.g. q1 with k0 via collect_vectors).

    Args:
        cache:                   ActivationCache from bridge.run_with_cache
                                 (use prepend_bos=False so indices match the map)
        multi_token_words_map:   output of tokenization.get_multi_token_words
        out_dir:                 directory for .pt files
        slots:                   which of q0,q1,k0,k1,v0,v1 to write (subset ok)
        prefix:                  optional filename prefix (e.g. "Pile-CC_")
        n_layers:                layer count; inferred from cache keys if None

    Returns:
        { slot: path } for files written, e.g.
          {"q0": ".../q0.pt", "q1": ".../q1.pt", ...}

        Example — contents of q1.pt (2nd subtoken's query vectors):
          {
            " the": {
              (0, 5): [tensor(128,), tensor(128,), ...],  # each occ at layer 0, head 5
              (0, 6): [...],
              ...
            },
            "Colo": {
              (12, 3): [tensor(128,), ...],  # Q at the 2nd subtoken of each "Colo" occ
              ...
            },
          }
        i.e. word → (layer, head) → list of (d_head,) Q vectors for subtoken index 1.
    """
    slots = tuple(slots)
    for s in slots:
        _parse_slot(s)

    # Which of q/k/v we need to pull from the cache (e.g. {"q","k"} if slots are q*/k*)
    roles_needed = { _parse_slot(s)[0] for s in slots }
    if n_layers is None:
        # blocks.{i}.attn.hook_rot_k or v.hook_out
        layer_idxs = set()
        for key in cache.keys():
            m = re.match(r"blocks\.(\d+)\.attn\.", key)
            if m:
                layer_idxs.add(int(m.group(1)))
        if not layer_idxs:
            raise ValueError("Could not infer n_layers from cache keys")
        n_layers = max(layer_idxs) + 1

    # Final output builder: by_slot["q1"][word][(layer, head)] = [occ vectors...]
    by_slot: dict[str, dict] = {s: {} for s in slots}

    for word, info in multi_token_words_map.items():
        # Per-word scratch pad while we walk occurrences:
        #   word_accums[slot][(layer, head)] = [vec_occ0, vec_occ1, ...]
        # Each vec is one (d_head,) Q/K/V vector for that slot at that occurrence.
        word_accums = {s: defaultdict(list) for s in slots}
        # Count of bitoken occurrences kept for this word (skip if zero so we
        # don't write an empty entry into by_slot).
        n_added = 0

        for occ in info["occurrences"]:
            token_indices = occ["token_indices"]
            if len(token_indices) != 2:
                continue
            # Sequence positions of the first and second subtokens
            i0, i1 = token_indices[0], token_indices[1]
            n_added += 1

            for layer_idx in range(n_layers):
                role_tensors = {
                    role: cache[f"blocks.{layer_idx}.attn.{_ROLE_HOOK[role]}"]
                    for role in roles_needed
                }
                # role_tensors[role]: [batch, seq, n_role_heads, d_head]
                # Index by the largest head count (usually Q) so q/k/v share
                # the same (layer, head) keys even under GQA.
                n_index_heads = max(t.shape[2] for t in role_tensors.values())

                for head_idx in range(n_index_heads):
                    layer_head = (layer_idx, head_idx)

                    for slot in slots:  # e.g. "k0" → role="k", subtoken=0
                        role, subtoken_idx = _parse_slot(slot)
                        token_pos = i0 if subtoken_idx == 0 else i1

                        # Full Q, K, or V activation for this layer from the TL cache:
                        # shape [batch, seq, n_role_heads, d_head].
                        activation = role_tensors[role]
                        n_role_heads = activation.shape[2]
                        # GQA: query head `head_idx` maps to the shared K/V head via
                        # modulo (role_head_idx = head_idx % n_kv_heads). Several query
                        # heads therefore read the same K/V. Olmo-3 7B has
                        # n_heads == n_kv_heads (32/32), so this is a no-op there.
                        role_head_idx = head_idx % n_role_heads

                        # Single head vector at this token: (d_head,) float32 on CPU.
                        vec = activation[0, token_pos, role_head_idx, :].float().cpu()
                        word_accums[slot][layer_head].append(vec)

        if n_added:
            for slot in slots:
                by_slot[slot][word] = dict(word_accums[slot])

    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for slot in slots:
        path = os.path.join(out_dir, f"{prefix}{slot}.pt")
        torch.save(by_slot[slot], path)
        paths[slot] = path
        print(f"[save_slot_vectors] Wrote {path} ({len(by_slot[slot])} words)")
    return paths


def collect_vectors(
    path_a: str,
    path_b: str,
    name_a: str,
    name_b: str,
) -> dict:
    """
    Load two slot files and merge into the paired analysis schema.

    Returns:
        { word: { (layer, head): { name_a: [occ...], name_b: [occ...] } } }
    """
    a = load_vectors(path_a)
    b = load_vectors(path_b)
    out = {}
    for word in a:
        if word not in b:
            continue
        out[word] = {
            lh: {name_a: a[word][lh], name_b: b[word][lh]}
            for lh in a[word]
            if lh in b[word]
        }
    return out


def save_vectors(collected: dict, path: str) -> None:
    """Save a paired (or any) collected dict to a .pt file."""
    torch.save(collected, path)


def load_vectors(path: str) -> dict:
    """Load a .pt file written by save_slot_vectors / save_vectors / collect_vectors."""
    return torch.load(path, map_location="cpu", weights_only=False)


def filter_collected(collected: dict, filter_name: str) -> dict:
    """
    Returns a subset of collected keyed only by words matching filter_name.

    Args:
        collected:   paired dict from collect_vectors
        filter_name: "all" to return unchanged, or any key from WORD_CATEGORIES
    """
    if filter_name == "all":
        return collected
    valid = set(WORD_CATEGORIES.keys())
    if filter_name not in valid:
        raise ValueError(f"Unknown filter {filter_name!r}. Valid: {sorted(valid)}")
    return {w: v for w, v in collected.items() if classify_word(w) == filter_name}


def micro_average_vectors(collected: dict, pair: tuple[str, str] | None = None) -> dict:
    """
    Micro-average: flatten all occurrences of all word types and mean together.

    Returns:
        { (layer, head): { slot_a: tensor(d,), slot_b: tensor(d,) } }
    """
    if not collected:
        return {}
    sample_word = next(iter(collected.values()))
    sample_inner = next(iter(sample_word.values()))
    name_a, name_b = pair if pair is not None else _pair_keys(sample_inner)

    accum = defaultdict(lambda: {name_a: [], name_b: []})
    for word_data in collected.values():
        for key, val in word_data.items():
            accum[key][name_a].extend(val[name_a])
            accum[key][name_b].extend(val[name_b])

    return {
        key: {
            name_a: torch.stack(val[name_a]).mean(dim=0),
            name_b: torch.stack(val[name_b]).mean(dim=0),
        }
        for key, val in accum.items()
    }


def macro_average_vectors(collected: dict, pair: tuple[str, str] | None = None) -> dict:
    """
    Macro-average: mean within each word type first, then across word types.
    """
    if not collected:
        return {}
    sample_word = next(iter(collected.values()))
    sample_inner = next(iter(sample_word.values()))
    name_a, name_b = pair if pair is not None else _pair_keys(sample_inner)

    accum = defaultdict(lambda: {name_a: [], name_b: []})
    for word_data in collected.values():
        for key, val in word_data.items():
            accum[key][name_a].append(torch.stack(val[name_a]).mean(dim=0))
            accum[key][name_b].append(torch.stack(val[name_b]).mean(dim=0))

    return {
        key: {
            name_a: torch.stack(val[name_a]).mean(dim=0),
            name_b: torch.stack(val[name_b]).mean(dim=0),
        }
        for key, val in accum.items()
    }


def compute_polar_per_head(
    averaged: dict,
    pair: tuple[str, str] | None = None,
) -> dict:
    """
    Polar coords (r, theta) per (layer, head). Second slot in pair is the reference.

    Args:
        averaged: output of micro/macro_average_vectors
        pair:     (name_a, name_b); inferred from dict keys if None

    Returns:
        { (layer, head): (r, theta) }
    """
    if not averaged:
        return {}
    sample = next(iter(averaged.values()))
    name_a, name_b = pair if pair is not None else _pair_keys(sample)
    return {
        lh: get_polar_coordinates(vecs[name_a], vecs[name_b])
        for lh, vecs in averaged.items()
    }


def plot_polar_grid(
    polar_data: dict,
    output_dir: str = "visuals/experiments/exp3_polar",
    nrows: int = 8,
    ncols: int = 4,
    title_suffix: str = "vector polar coords",
) -> None:
    """
    For each layer, plots a grid of polar plots — one subplot per head.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not polar_data:
        print("[plot_polar_grid] No data to plot.")
        return

    num_heads = nrows * ncols
    present_layers = sorted({layer for layer, _ in polar_data})

    for layer in present_layers:
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

        fig.suptitle(f"Layer {layer} — {title_suffix}", fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = os.path.join(output_dir, f"layer_{layer}_polar_grid.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

    print(f"Done — {len(present_layers)} layer plots saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    from model import get_bridge
    from tokenization import get_multi_token_words
    from utils import load_json

    json_path = sys.argv[1] if len(sys.argv) > 1 else "output/multi_word_output.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/qkv_slots"
    in_map = load_json(json_path)
    text = in_map["text"]
    multi_token_words_map = in_map["main_data"]

    bridge, tokenizer = get_bridge()
    tokens = bridge.to_tokens(text, prepend_bos=False)
    with torch.no_grad():
        _, cache = bridge.run_with_cache(tokens, prepend_bos=False)

    save_slot_vectors(cache, multi_token_words_map, out_dir)
    collected = collect_vectors(
        os.path.join(out_dir, "k0.pt"),
        os.path.join(out_dir, "k1.pt"),
        "k0",
        "k1",
    )
    averaged = micro_average_vectors(collected)
    polar_data = compute_polar_per_head(averaged)
    plot_polar_grid(polar_data)
