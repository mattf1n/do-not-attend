import torch
from collections import defaultdict
from utils import load_json

def aggregate_multi_token_word_attentions(attentions, multi_word_map):
    """
    Memory-efficient drop-in for aggregate_multi_token_word_attentions.
    Processes one layer at a time to avoid stacking all L layers into a single
    (L, H, S, S) tensor, which would double peak CPU memory.

    Within each layer, column indexing is still vectorized across heads.

    Input / output schema: same as the old version.

    Output shape:
      {
        "<word>": {
          "occurrences": [
            {
              "token_indices": [int, ...],
              "attentions":  {
                                "layers": [
                                    { "heads": [head_0_tensor, head_1_tensor, ...] },...
                                    ]
                            }
            },
            ...
          ]
        },
        ...
      }

    Where each head tensor has shape [num_subtokens, num_valid_rows]
    (num_valid_rows = seq_len - (max(target_positions) + 1)).
    """
    if not attentions:
        return {}
    if attentions[0].device.type != "cpu":
        attentions = tuple(a.detach().cpu() for a in attentions)

    # Pre-compute per-occurrence positions and valid-row starts once.
    occ_meta = {}
    for word, info in multi_word_map.items():
        if not (isinstance(info, dict) and "occurrences" in info):
            raise ValueError(
                f"Expected AI-friendly tokenization schema for word '{word}', but got: {info}"
            )
        occ_meta[word] = [
            {
                "token_indices": occ["token_indices"],
                "pos": torch.as_tensor(occ["token_indices"], dtype=torch.long),
                "start": int(torch.as_tensor(occ["token_indices"]).max()) + 1, # starting row
            }
            for occ in info["occurrences"]
        ]

    # layers_data[word][occ_idx] = list of per-layer head tensors
    layers_data = {
        word: [[] for _ in occs] for word, occs in occ_meta.items()
    }

    for layer_tensor in attentions:
        # shape: (H, S, S) — squeeze the batch dim, process, then release
        layer = layer_tensor[0]  # (H, S, S)
        for word, occs in occ_meta.items():
            for i, occ in enumerate(occs):
                pos, start = occ["pos"], occ["start"]
                # (H, num_valid_rows, K) -> transpose -> (H, K, num_valid_rows)
                block = layer[:, start:, pos].transpose(1, 2)
                layers_data[word][i].append({"heads": list(block.unbind(0))})
        del layer

    out = {}
    for word, occs in occ_meta.items():
        occurrences = []
        for i, occ in enumerate(occs):
            occurrences.append({
                "token_indices": occ["token_indices"],
                "attentions": {"layers": layers_data[word][i]},
            })
        out[word] = {"occurrences": occurrences}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NEW EXPERIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def get_biword_score_pairs(path: str) -> dict:
    """
    Returns { (layer_idx, head_idx): [(s0_val, s1_val), ...] } for every
    bi-token word occurrence in the output JSON.
    """
    result = defaultdict(list)
    mw_map = load_json(path)["main_data"]
    for word in mw_map:
        for occurrence in mw_map[word]["occurrences"]:
            for layer_idx, layer in enumerate(occurrence["attentions"]["layers"]):
                for head_idx, scores in enumerate(layer["heads"]):
                    if len(scores) == 2:
                        result[(layer_idx, head_idx)].extend(zip(scores[0], scores[1]))
    return dict(result)


def get_biword_score_pairs_diff(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each bi-token word occurrence, computes tok_1[row] - tok_0[row] for
    every shared row (i.e., every subsequent token position that attends to
    both subtokens).

    Returns:
        { (layer_idx, head_idx): [diff, ...] }
        where each diff is in [-1, 1] (positive = tok_1 heavier, negative = tok_0 heavier).
    """
    pairs = get_biword_score_pairs(path)
    return {
        key: [s1 - s0 for s0, s1 in vals]
        for key, vals in pairs.items()
    }


def get_biword_score_pairs_contrast(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each bi-token word occurrence, computes the Michelson contrast per row:
        (tok_1 - tok_0) / (tok_1 + tok_0)
    This is scale-invariant: a small absolute diff at low values is weighted
    more heavily than the same diff at high values.

    Edge case: when tok_1 + tok_0 == 0, contrast is defined as 0.

    Returns:
        { (layer_idx, head_idx): [contrast, ...] }
        where each contrast is in [-1, 1].
    """
    pairs = get_biword_score_pairs(path)
    result = {}
    for key, vals in pairs.items():
        contrasts = []
        for s0, s1 in vals:
            denom = s0 + s1
            contrasts.append((s1 - s0) / denom if denom != 0 else 0.0)
        result[key] = contrasts
    return result


def compute_layer_contrast_means(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each layer, computes the mean Michelson contrast across all heads.

    Returns:
        { layer_idx: float }  — values in [-1, 1]
    """
    from collections import defaultdict
    contrasts = get_biword_score_pairs_contrast(path)
    layer_vals = defaultdict(list)
    for (layer, _), vals in contrasts.items():
        if vals:
            layer_vals[layer].append(sum(vals) / len(vals))
    return {layer: sum(vals) / len(vals) for layer, vals in sorted(layer_vals.items())}


def compute_head_hypothesis_rates(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each (layer, head), computes the fraction of row-level diffs where the
    last subtoken had strictly higher attention than the first (diff > 0).

    Returns:
        { (layer_idx, head_idx): float }  — values in [0, 1]
    """
    diffs = get_biword_score_pairs_diff(path)
    return {
        #NOTE: v = s1 - s0; therefore v > 0 means s1 > s0
        head: sum(1 for v in vals if v > 0) / len(vals)
        for head, vals in diffs.items()
    }


def compute_layer_hypothesis_rates(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each layer, computes the mean hypothesis rate across all heads.

    Returns:
        { layer_idx: float }  — values in [0, 1]
    """
    from collections import defaultdict
    head_rates = compute_head_hypothesis_rates(path)
    layer_vals = defaultdict(list)
    for (layer, _), rate in head_rates.items():
        layer_vals[layer].append(rate)
    return {layer: sum(vals) / len(vals) for layer, vals in sorted(layer_vals.items())}


def summarize_hypothesis_coverage(
    rates: dict,
    threshold: float = 0.5,
) -> tuple:
    """
    Given per-(layer, head) hypothesis rates, returns which heads pass the threshold
    and the overall percentage of heads that do.

    Args:
        rates:     output of compute_head_hypothesis_rates (coming from diff)
        threshold: minimum rate to count a head as "following" the hypothesis

    Returns:
        (passing_heads, overall_pct)
        - passing_heads: list of (layer, head) tuples with rate >= threshold
        - overall_pct:   float in [0, 100]
    """
    passing = [(k, v) for k, v in rates.items() if v >= threshold]
    passing_heads = [k for k, _ in sorted(passing)]
    overall_pct = 100.0 * len(passing_heads) / len(rates) if rates else 0.0
    return passing_heads, overall_pct
