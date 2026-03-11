import torch
from config import DEVICE
import copy


def get_multi_token_word_attentions(attentions, target_positions):
    """
    For given token positions (subtokens of a multi-token word), gathers the attention columns for each position,
    for all layers and heads, only considering rows after the last target position (where a target token is NOT the last non-masked token in a row).

    Args:
        attentions (Iterable[Tensor]): List or tuple of attention tensors per layer from the model
            (each tensor shape: [batch, heads, seq_len, seq_len]).
        target_positions (List[int]): The sequence indices of the target subtokens within the sequence.

    Returns:
        dict: Maps each target position to a nested dict of attentions[layer][head] = attention_column_tensor,
            where each attention_column_tensor contains a single subtoken's attention scores from valid_start:end.
    """
    attentions = torch.stack(attentions)[:, 0]  # (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = attentions.shape
    last_pos = max(target_positions)
    valid_start = last_pos + 1  # only include attention rows after the last target_position

    target_columns = [] #not using tensor b/c each col will be different length
    for pos in target_positions:
        pos_columns = {}
        for layer in range(num_layers):
            pos_columns[layer] = {}
            for head in range(num_heads):
                col = attentions[layer, head, valid_start:, pos]
                pos_columns[layer][head] = col
        target_columns.append(pos_columns)

    return target_columns




def aggregate_multi_token_word_attentions(attentions, multi_word_map):
    """
    Compute attention analysis for every multi-token word in the text.
    Args:
        attentions: tuple of attention tensors from model output, one per layer.
        multi_word_map: output of get_multi_token_words, mapping words to tokens/positions.
    Returns:
        dict mapping each multi-token word to its per-subtoken attention stats.
    """
    mw_map = multi_word_map.copy()
    for word, info in mw_map.items():
        mw_map[word]['position_attentions'] = {}
        positions = mw_map[word]['positions']
        for i, position_tuple in positions.items():
            mw_map[word]['position_attentions'][i] = get_multi_token_word_attentions(attentions, position_tuple)

    return mw_map



def get_multi_token_word_attentions_ai(attentions, target_positions):
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


def aggregate_multi_token_word_attentions_ai(attentions, multi_word_map):
    """
    AI-friendly schema: bundle each occurrence's token span with its attention tensors.

    Output shape:
      {
        "<word>": {
          "occurrences": [
            {
              "token_indices": [int, ...],
              "attentions": <return value of get_multi_token_word_attentions_ai(...) for that occurrence>
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
                    "attentions": get_multi_token_word_attentions_ai(attentions, token_positions),
                }
            )
        out[word] = {"occurrences": occurrences}

    return out
