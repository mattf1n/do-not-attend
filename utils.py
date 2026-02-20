import re
import string
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(dataset_name="monology/pile-uncopyrighted"):
    data = load_dataset(dataset_name, streaming=True)
    return data

def get_data_model():
    data = get_data()
    model, tokenizer = get_model()
    return (data, model, tokenizer)

def get_model(model_name="allenai/Olmo-3-1025-7B"):
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_data_sample():
    data = get_data()
    sample = next(iter(data['train']))
    return sample['text']


def get_first_paragraph():
    data = get_data()
    sample = next(iter(data['train']))
    return sample['text'].split("\n\n")[0] #gets the first paragraph

def get_multi_token_words(text, tokenizer):
    '''
    finds all multi-token words

    exception: if the last word in sentence is multi-token we exclude it
    '''

    result = {}
    words = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)
    pos = 0
    total_tokens = len(tokenizer.tokenize(text))

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        n = len(word_tokens)

        if word.isalpha() and n > 1 and pos + n < total_tokens:
            result[word] = {
                "tokens": word_tokens,
                "positions": list(range(pos, pos + n)),
            }
        pos += n
    return result



def get_attentions(input_data, model, tokenizer):
    '''
    from input string, run inference and get the attention scores of the model
    '''
    inputs = tokenizer(input_data, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions


def p1_att():
    '''
    run model on first paragraph of data. 
    returns: attention scores
    '''

    text = get_first_paragraph()
    model, tokenizer = get_model()
    att = get_attentions(text, model, tokenizer)
    return att, text


def clean_subtokens(text, positions, tokenizer):

    def clean_token(tok, tokenizer):
        ids = tokenizer.convert_tokens_to_ids([tok])
        return tokenizer.decode(ids).strip()

    raw_tokens = tokenizer.tokenize(text)
    return [clean_token(raw_tokens[p], tokenizer) for p in positions]


def analyze_multi_token_attention(attentions, target_positions, token_strings=None):
    all_attn = torch.stack(attentions)[:, 0]
    num_layers, num_heads, seq_len, _ = all_attn.shape

    last_pos = max(target_positions)
    valid_start = last_pos + 1

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
        results[word] = analyze_multi_token_attention(
            attentions, info["positions"], token_strings=info["tokens"]
        )
    return results

def summarize_results(results,tokenizer):
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


def test_pipeline():
    """End-to-end pipeline: load data/model, run inference, analyze attention.

    Returns:
        paragraph: the input text
        multi: dict of multi-token words with their tokens and positions
        results: per-word, per-subtoken attention analysis
        summary: condensed mapping of subtoken strings to max attention scores
    """

    choice = input("1: paragraph \n2: one doc ")

    model, tokenizer = get_model()

    if (choice == "1"):
        print("paragraph selected")
        text = get_first_paragraph()
    elif (choice == "2"):
        print("one doc selected")
        text = get_data_sample()
    else:
        print("default selected")
        text = get_first_paragraph()

   
    attentions = get_attentions(text, model, tokenizer)
    multi = get_multi_token_words(text, tokenizer)

    results = get_all_attention_results(attentions, multi, text, tokenizer)
    max_summary = summarize_results(results, tokenizer)

    return text, multi, results, max_summary




if __name__ == "__main__":  
    text, multi, results, max_summary = test_pipeline()

    pprint(max_summary)