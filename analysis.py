import torch
from config import DEVICE
import copy

# You also need to import get_first_paragraph and get_model (source not shown)
# Example:
# from utils import get_first_paragraph, get_model

def p1_att():
    '''
    run model on first paragraph of data. 
    returns: attention scores
    '''
    text = get_first_paragraph()
    model, tokenizer = get_model()
    att = get_attentions(text, model, tokenizer)
    return att, text


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





def find_num_tokens_each_word(json_path="sample_results.json"):


    with open("sample_results.json") as f:
        data = json.load(f)

    sizes = Counter(len(v) for v in data.values())
    print(sizes)



def summarize_results_max(results, tokenizer):
    """
    For each multi-token word, maps each subtoken's cleaned string
    to its max attention score.

    Args:
        results: output of test_pipeline's results dict
    """
    summary = {}
    for word, subtokens in results.items():
        word_summary = {}
        for i, info in subtokens.items():
            clean = tokenizer.decode(tokenizer.convert_tokens_to_ids([info["token"]])).strip()
            word_summary[clean] = info["max_attention"]
        summary[word] = word_summary
    return summary


