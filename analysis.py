import torch
from config import DEVICE

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


def find_num_tokens_each_word(json_path="sample_results.json"):


    with open("sample_results.json") as f:
        data = json.load(f)

    sizes = Counter(len(v) for v in data.values())
    print(sizes)


def get_attentions(input_data, model, tokenizer):
    '''
    from input string, run inference and get the attention scores of the model
    '''
    inputs = tokenizer(input_data, return_tensors="pt")
    # Move inputs to DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions




def analyze_multi_token_attention_max(attentions, target_positions, token_strings=None):
    all_attn = torch.stack(attentions)[:, 0] #fixes the shape to be all tensor (as opposed to tuple(tensor))
    num_layers, num_heads, seq_len, _ = all_attn.shape

    last_pos = max(target_positions)
    valid_start = last_pos + 1 # will dictate where we start (rows that dont have word as the last step)

    results = {}

    for i, pos in enumerate(target_positions):
        global_max = -1.0
        global_max_loc = None
        head_maxes = []

        for layer in range(num_layers):
            for head in range(num_heads):
                col = all_attn[layer, head, valid_start:, pos]

                if col.numel() == 0:
                    head_maxes.append(0.0)
                    continue

                head_max = col.max().item()
                head_maxes.append(head_max)

                if head_max > global_max:
                    global_max = head_max
                    row_idx = valid_start + col.argmax().item()
                    global_max_loc = (layer, head, row_idx)

        results[i] = {
            "token": token_strings[i] if token_strings else str(pos),
            "position": pos,
            "max_attention": global_max,
            "max_location (layer, head, row)": global_max_loc,
            "avg_max_per_head": sum(head_maxes) / len(head_maxes),
        }

    return results


def get_multi_token_attentions(attentions, target_positions, token_strings=None):
    """
    For a list of target positions (i.e., subtokens of a multi-token word),
    collects the attention columns for each subtoken, for every layer and head.

    Returns:
        results: dict mapping subtoken string (or index) to:
            dict mapping layer to dict mapping head to attention column tensor
    """
    all_attn = torch.stack(attentions)[:, 0]  # (num_layers, num_heads, seq_len, seq_len)
    num_layers, num_heads, seq_len, _ = all_attn.shape

    last_pos = max(target_positions)
    valid_start = last_pos + 1  # only include attentions after the word

    results = {}
    names = token_strings if token_strings is not None else list(map(str, target_positions))
    for pos_name, pos in zip(names, target_positions):
        results[pos_name] = {}
        for layer in range(num_layers):
            results[pos_name][layer] = {}
            for head in range(num_heads):
 
                col = all_attn[layer, head, valid_start:, pos]
                results[pos_name][layer][head] = col

    return results



def get_all_attention_results(attentions, multi, text, tokenizer):
    """Compute attention analysis for every multi-token word in the text.

    Args:
        attentions: tuple of attention tensors from model output, one per layer
        multi: output of get_multi_token_words, mapping words to tokens/positions
        text: the original input string (used for cleaning subtoken strings)
        tokenizer: the tokenizer used for encoding

    Returns:
        dict mapping each multi-token word to its per-subtoken attention stats
    """
    results = {}
    for word, info in multi.items():
        results[word] = get_multi_token_attentions(
            attentions, info["positions"], token_strings=info["tokens"]
        )
    return results


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


