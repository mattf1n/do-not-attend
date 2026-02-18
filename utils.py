import re
import string
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    result = {}
    words = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)
    pos = 0

    for word in words:
        word_tokens = tokenizer.tokenize(word)
        n = len(word_tokens)

        if word.isalpha() and len(word_tokens) > 1:
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




def analyze_multi_token_attention(text, model, tokenizer, top_k=None):
    """Analyze attention patterns for multi-token words.

    For each multi-token word, finds:
    - max_per_head: top-k attended tokens per layer/head combo
    - max_global: majority vote across all layers/heads (top 5)

    Args:
        text: input string to analyze
        model: HuggingFace model with eager attention
        tokenizer: corresponding tokenizer
        top_k: number of top tokens to track per word. Defaults to
               the number of subtokens in each word.

    Returns:
        dict mapping each multi-token word to its attention results
    """
    from collections import Counter

    

    multi = get_multi_token_words(text, tokenizer)
    att = get_attentions(text, model, tokenizer)
    att = [a.squeeze(0) for a in att]  # remove batch dim
    raw_tokens = tokenizer.tokenize(text)


    def clean(tok):
        """Convert BPE token to readable string."""
        ids = tokenizer.convert_tokens_to_ids([tok])
        return tokenizer.decode(ids).strip()

    tokens = [clean(t) for t in raw_tokens]  # clean once

    num_layers = len(att)
    num_heads = att[0].shape[0]

    results = {}

    for word, info in multi.items():
        positions = info["positions"]
        k = top_k if top_k is not None else len(positions)
        clean_subtokens = [tokens[p] for p in positions]

        word_results = {
            "tokens": clean_subtokens,
            "positions": positions,
            "top_k": k,
            "max_per_head": {},
            "max_global": {},
        }

        for i, pos in enumerate(positions):
            head_scores = []
            vote_counts = Counter()

            for layer in range(num_layers):
                for head in range(num_heads):
                    row = att[layer][head][pos].clone()
                    row[pos] = 0.0  # exclude self-attention

                    top_vals, top_idxs = row.topk(k)
                    top_tokens = [(tokens[idx], top_vals[j].item()) for j, idx in enumerate(top_idxs)]

                    head_scores.append({
                        "layer": layer,
                        "head": head,
                        "top_k": top_tokens,
                    })

                    max_idx = row.argmax().item()
                    vote_counts[tokens[max_idx]] += 1

            word_results["max_per_head"][clean_subtokens[i]] = head_scores
            total_votes = num_layers * num_heads
            word_results["max_global"][clean_subtokens[i]] = [
                (token, count, f"{count/total_votes:.1%}")
                for token, count in vote_counts.most_common(5)
            ]

        results[word] = word_results

    return results


def test_pipeline():
    '''
    for one setence, 
        - find all of the multi-token words
        - analyze the max attention scores of said words
    '''

    paragraph = get_first_paragraph()

    model,tokenizer = get_model()

    multi = get_multi_token_words(paragraph, tokenizer)

    results = analyze_multi_token_attention(paragraph, model, tokenizer, top_k=None)

    return paragraph, multi, results
    