import torch
from config import DEVICE
import copy

def get_multi_token_word_attentions(attentions, target_positions):
    """
    AI-friendly variant of get_multi_token_word_attentions.

    For a single multi-token word (described by its subtoken indices in `target_positions`),
    returns a nested structure that can be easily serialized to JSON:

      {
        "token_indices": [11, 12],
        "attentions": {
          "layers": [
            { "heads": [head_0_tensor, head_1_tensor, ...] },
            ...
          ]
        }
      }

    Where each head tensor has shape [num_subtokens, num_valid_rows]
    (num_valid_rows = seq_len - (max(target_positions) + 1)).
    """
    atts = torch.stack(attentions)[:, 0]  # (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = atts.shape

    last_pos = max(target_positions)
    valid_start = last_pos + 1
    token_positions = list(target_positions)

    layers_out = []
    for layer in range(num_layers):
        heads_out = []
        for head in range(num_heads):
            cols = []
            for pos in token_positions:
                col = atts[layer, head, valid_start:, pos]  # (num_valid_rows,)
                cols.append(col)
            if len(cols) == 1:
                head_tensor = cols[0].unsqueeze(0)  # [1, num_valid_rows]
            else:
                head_tensor = torch.stack(cols, dim=0)  # [num_subtokens, num_valid_rows]
            heads_out.append(head_tensor)
        layers_out.append({"heads": heads_out})

    # return {
    #     "token_indices": token_positions,
    #     "attentions": {"layers": layers_out},
    # }
    out = {"layers": layers_out}
    return out


def aggregate_multi_token_word_attentions(attentions, multi_word_map):
    """
    AI-friendly schema: bundle each occurrence's token span with its attention tensors.

    Output shape:
      {
        "<word>": {
          "occurrences": [
            {
              "token_indices": [int, ...],
              "attentions": <return value of get_multi_token_word_attentions(...) for that occurrence>
            },
            ...
          ]
        },
        ...
      }
    """
    out = {}
    for word, info in multi_word_map.items():
        occurrences = []
        # Only accepts the AI-friendly tokenization schema (must have "occurrences" key)
        if not (isinstance(info, dict) and "occurrences" in info):
            raise ValueError(f"Expected AI-friendly tokenization schema for word '{word}', but got: {info}")

        for occ in info["occurrences"]:
            token_positions = occ["token_indices"]
            occurrences.append(
                {
                    "token_indices": list(token_positions),
                    "attentions": get_multi_token_word_attentions(attentions, token_positions),
                }
            )
        out[word] = {"occurrences": occurrences}

    return out
