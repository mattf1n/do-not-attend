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

    Where each head tensor has shape [num_subtokens] — the mean attention
    score across all valid attending rows, one value per subtoken.
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
                # (H, num_valid_rows, K) -> mean over rows -> (H, K)
                block_mean = layer[:, start:, pos].mean(dim=1)
                layers_data[word][i].append(block_mean)
        del layer

    out = {}
    for word, occs in occ_meta.items():
        occurrences = []
        for i, occ in enumerate(occs):
            layer_dicts = [{"heads": list(t.unbind(0))} for t in layers_data[word][i]]
            occurrences.append({
                "token_indices": occ["token_indices"],
                "attentions": {"layers": layer_dicts},
            })
        out[word] = {"occurrences": occurrences}
    return out

def aggregate_multi_token_word_attentions_old(attentions, multi_word_map):
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

    Where each head tensor has shape [num_subtokens] — the mean attention
    score across all valid attending rows, one value per subtoken.
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
                # (H, num_valid_rows, K) -> transpose -> (H, K, num_valid_rows) -> mean -> (H, K)
                block = layer[:, start:, pos].transpose(1, 2)
                block_mean = block.mean(dim=2)
                layers_data[word][i].append({"heads": list(block_mean.unbind(0))})
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
# STREAMING AGGREGATOR (layer-at-a-time)
# ─────────────────────────────────────────────────────────────────────────────


class MultiTokenWordAggregator:
    """
    Streaming variant of aggregate_multi_token_word_attentions.

    Designed to be driven from a forward hook on each decoder layer's
    `self_attn` module: feed one layer's attention tensor at a time, the
    aggregator reduces it to per-occurrence/per-head/per-subtoken means
    immediately, and the raw [H, S, S] tensor can be freed before the next
    layer runs.

    Output schema (from `finalize`) matches `aggregate_multi_token_word_attentions`.

    Typical usage:
        agg = MultiTokenWordAggregator(multi_word_map)
        # in a forward hook on layer i:
        #     agg.add_layer(attn_weights)        # attn_weights: (1, H, S, S)
        result = agg.finalize()
    """

    def __init__(self, multi_word_map):
        self.multi_word_map = multi_word_map
        self.occ_meta = {}
        for word, info in multi_word_map.items():
            if not (isinstance(info, dict) and "occurrences" in info):
                raise ValueError(
                    f"Expected AI-friendly tokenization schema for word '{word}', "
                    f"but got: {info}"
                )
            self.occ_meta[word] = [
                {
                    "token_indices": occ["token_indices"],
                    "pos": torch.as_tensor(occ["token_indices"], dtype=torch.long),
                    "start": int(torch.as_tensor(occ["token_indices"]).max()) + 1,
                }
                for occ in info["occurrences"]
            ]
        self.layers_data = {
            word: [[] for _ in occs] for word, occs in self.occ_meta.items()
        }

    def add_layer(self, layer_tensor):
        """
        Reduce one layer's attention to per-occurrence (H, K) means and append.

        layer_tensor: torch.Tensor of shape (1, H, S, S) or (H, S, S), any
        device/dtype. The reduction runs on whatever device the tensor is on,
        so passing a CUDA tensor keeps the heavy gather on GPU and only the
        small (H, K) result is moved to CPU.
        """
        layer = layer_tensor[0] if layer_tensor.dim() == 4 else layer_tensor
        device = layer.device
        for word, occs in self.occ_meta.items():
            for i, occ in enumerate(occs):
                pos = occ["pos"].to(device)
                start = occ["start"]
                # (H, num_valid_rows, K) -> mean over rows -> (H, K)
                block = layer[:, start:, pos].transpose(1, 2)
                block_mean = block.mean(dim=2)
                block_mean = block_mean.detach().to("cpu", torch.float16)
                self.layers_data[word][i].append(
                    {"heads": list(block_mean.unbind(0))}
                )

    def finalize(self):
        out = {}
        for word, occs in self.occ_meta.items():
            occurrences = [
                {
                    "token_indices": occ["token_indices"],
                    "attentions": {"layers": self.layers_data[word][i]},
                }
                for i, occ in enumerate(occs)
            ]
            out[word] = {"occurrences": occurrences}
        return out


def get_attentions_streaming(text, model, tokenizer, multi_word_map):
    """
    Run the model and aggregate multi-token word attentions in a single pass,
    one layer at a time, without ever holding the full L-layer attention tuple
    in memory.

    Memory ceiling: peak ~ one layer's [H, S, S] tensor instead of L of them.
    For OLMo-3-7B on CPU this raises the practical context-length ceiling
    from ~7k to ~45k tokens at 150 GB free RAM.

    Returns the same dict schema as `aggregate_multi_token_word_attentions`.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    agg = MultiTokenWordAggregator(multi_word_map)

    def make_hook(layer_idx):
        def hook(module, args, outputs):
            if not (isinstance(outputs, tuple) and len(outputs) > 1):
                return outputs
            attn = outputs[1]
            if attn is None:
                return outputs
            agg.add_layer(attn)
            # Replace HF's reference to the raw attention tensor so the model
            # doesn't accumulate all L of them to return as a tuple.
            return (outputs[0], None) + outputs[2:]
        return hook

    n = model.config.num_hidden_layers
    handles = [
        layer.self_attn.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers[:n])
    ]
    try:
        with torch.no_grad():
            model(**inputs, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    return agg.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# NEW EXPERIMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def get_biword_score_pairs(path: str) -> dict:
    """
    Returns { (layer_idx, head_idx): [(s0_val, s1_val), ...] } for every
    bi-token word occurrence in the output JSON.

    Each (s0, s1) pair is the mean attention score across all attending rows
    for that occurrence — one pair per occurrence, not one per attending row.
    """
    result = defaultdict(list)
    mw_map = load_json(path)["main_data"]
    for word in mw_map:
        for occurrence in mw_map[word]["occurrences"]:
            for layer_idx, layer in enumerate(occurrence["attentions"]["layers"]):
                for head_idx, scores in enumerate(layer["heads"]):
                    if len(scores) == 2:
                        # scores[0] and scores[1] are per-occurrence averages, not row vectors
                        result[(layer_idx, head_idx)].append((scores[0], scores[1]))
    return dict(result)


def get_biword_score_pairs_diff(
    path: str = "output/multi_word_output.json",
) -> dict:
    """
    For each bi-token word occurrence, computes mean(tok_1) - mean(tok_0),
    where each score is the mean attention across all attending rows for that occurrence.

    Returns:
        { (layer_idx, head_idx): [diff, ...] }
        where each diff is one value per occurrence (positive = tok_1 heavier, negative = tok_0 heavier).
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
    For each bi-token word occurrence, computes the Michelson contrast using
    per-occurrence averaged subtoken scores:
        (mean(tok_1) - mean(tok_0)) / (mean(tok_1) + mean(tok_0))
    This is scale-invariant: a small absolute diff at low values is weighted
    more heavily than the same diff at high values.

    Edge case: when tok_1 + tok_0 == 0, contrast is defined as 0.

    Returns:
        { (layer_idx, head_idx): [contrast, ...] }
        where each contrast is one value per occurrence, in [-1, 1].
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
    For each (layer, head), computes the fraction of occurrences where the
    last subtoken's mean attention is strictly higher than the first (diff > 0).

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
