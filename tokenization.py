from collections import defaultdict

def get_multi_token_words(text, tokenizer, max_num_subtokens=2):
    """
    Find all words that are split into multiple subtokens by the tokenizer,
    and record every position where each such word occurs in the token sequence.

    A word qualifies if its subtoken count is between 2 and max_num_subtokens
    (inclusive). Each occurrence is recorded separately so callers can look up
    attention scores at the exact token indices for that instance.

    Args:
        text: raw input string to tokenize.
        tokenizer: HuggingFace tokenizer that supports word_ids().
        max_num_subtokens: upper bound on subtoken count (default 2 = bitoken only).

    Returns:
        {
          "<word>": {
            "occurrences": [
              { "token_indices": [int, ...] },
              ...
            ]
          },
          ...
        }
    """
    encodings = tokenizer(text)
    #word_ids: word corresponding to each token
    word_ids = encodings.word_ids()   # maps token position → word index (None for special tokens)
    input_ids = encodings["input_ids"]

    # Single O(S) pass: group token positions by their word_id.
    # word_ids.index() inside a loop would be O(W×S); this avoids that.
    word_id_to_indices = defaultdict(list)
    for tok_idx, word_id in enumerate(word_ids):
        if word_id is not None:  # skip special tokens ([CLS], [SEP], etc.)
            word_id_to_indices[word_id].append(tok_idx)

    out = {}
    for word_id, indices in word_id_to_indices.items():
        count = len(indices)
        if 2 <= count <= max_num_subtokens:
            word = tokenizer.decode([input_ids[i] for i in indices])
            if word not in out:
                out[word] = {"occurrences": []}
            out[word]["occurrences"].append({"token_indices": indices}) #implicity takes into accoutn having multiple instances of the same word

    return out



def export_multi_token_words_map(multi_word_map, output_path):
    """
    Export the multi_token words map to a json file.

    Args:
        multi_word_map: The dictionary produced by get_multi_token_words.
        output_path: Path (str) to the json file for export.
    """
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(multi_word_map, f, ensure_ascii=False, indent=2)

def summarize_multi_token_words(multi_token_words_map):
    """
    Print and return the number of unique multi-token words and their total occurrences.

    Args:
        multi_token_words_map: output of get_multi_token_words

    Returns:
        (num_words, total_occurrences): tuple of ints
    """
    num_words = len(multi_token_words_map)
    total_occurrences = sum(len(info["occurrences"]) for info in multi_token_words_map.values())
    print(f"Unique multi-token words: {num_words}")
    print(f"Total occurrences:        {total_occurrences}")
    return num_words, total_occurrences



