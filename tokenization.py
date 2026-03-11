import json
import re
import numpy as np
from collections import Counter

def get_multi_token_words(text, tokenizer, max_num_subtokens=100):
    """
    AI-friendly schema: represent each multi-token word as a list of occurrences,
    where each occurrence bundles the subtoken span indices.

    Output shape:
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
    word_ids = encodings.word_ids()
    input_ids = encodings["input_ids"]

    num_token_count = Counter(word_ids)
    out = {}

    for word_id, count in num_token_count.items():
        if 2 <= count <= max_num_subtokens:
            start_idx = word_ids.index(word_id)
            end_idx = start_idx + count
            word_input_ids = [input_ids[i] for i in range(start_idx, end_idx)]
            word = tokenizer.decode(word_input_ids)
            token_indices = list(range(start_idx, end_idx))

            if word not in out:
                out[word] = {"occurrences": []}
            out[word]["occurrences"].append({"token_indices": token_indices})

    return out



