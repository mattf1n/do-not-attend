import json
import re
import numpy as np
from collections import Counter

def get_multi_token_words(text,tokenizer, max_num_tokens = 100 ):
    """
    Identifies all multi-token words in the input text using the given tokenizer.

    Args:
        text (str): The input text to analyze.
        tokenizer: The tokenizer object with a compatible encode/tokenize interface.
        max_num_tokens (int): Maximum number of tokens to consider for a word (not used in function body).

    Returns:
        list: A list of arrays, each containing the tokens of a multi-token word.
    """    
    encodings = tokenizer(text)
    word_ids = encodings.word_ids() #     #word_id: gives word id for each index of the token\
    # tokens = np.array(encodings.tokens())
    input_ids = encodings['input_ids']

    num_token_count = Counter(word_ids)

    multi_token_word_map = {}
    for word_id, count in num_token_count.items():
        if (2 <= count <= max_num_tokens):  # only look at multi-token words in specified range
            start_idx = word_ids.index(word_id) #find first index that uses word
            end_idx = start_idx + count

            word_input_ids = [input_ids[i] for i in range(start_idx, end_idx)]
            word = tokenizer.decode(word_input_ids)
            # word_tokens = tokens[start_idx: end_idx] # find the tokens that make up the multi-token word
            positions = list(range(start_idx, end_idx))

            
            if word in multi_token_word_map: #add new position for word
                #reason? Because there may be multiplt instances of the same word in the text
                multi_token_word_map[word]['positions'].append(positions)
            else: # add word + info for one position
                multi_token_word_map[word] = {
                    # 'tokens': word_tokens,
                    'input_ids': word_input_ids,
                    'positions': [ positions ]
                }

    return multi_token_word_map



