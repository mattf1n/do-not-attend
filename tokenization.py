import json
import re
import numpy as np
from collections import Counter

def token_at(pos, text, tokenizer):
    tokens = tokenizer.tokenize(text)
    word = tokenizer.decode(tokenizer.convert_tokens_to_ids([tokens[pos]])).strip()
    print(word)


#TODO fix this funciton (almost done)
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
    #word_id: gives word id for each index of the token\
    word_ids = encodings.word_ids()
    tokens = encodings.tokens()
    tokens = np.array(tokens)
    input_ids = encodings['input_ids']

    #count how many tokens correspond to each word
    num_token_count = Counter(word_ids)

    multi_token_words = []
    for word_id, count in num_token_count.items():
        if (2 <= count < max_num_tokens):  # only look at multi-token words in specified range
            # print(count)
            #mt_word: multi-token word
            start_idx = word_ids.index(word_id) #find first index that uses word
            end_idx = start_token + count

            start_input_id = input_ids[start_idx]
            end_input_id = input_ids[end_idx]
            print(start_input_id, end_input_id)
            # print(start_token, end_token)
            multi_token_word_tokens = tokens[start_idx: end_idx] # find the tokens that make up the multi-token word

            multi_token_words.append(multi_token_word)

    return multi_token_words




def get_multi_token_words_old(text, tokenizer):
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

def clean_subtokens(text, positions, tokenizer):

    def clean_token(tok, tokenizer):
        ids = tokenizer.convert_tokens_to_ids([tok])
        return tokenizer.decode(ids).strip()

    raw_tokens = tokenizer.tokenize(text)
    return [clean_token(raw_tokens[p], tokenizer) for p in positions]

def find_num_tokens_each_word(json_path="sample_results.json"):


    with open("sample_results.json") as f:
        data = json.load(f)

    sizes = Counter(len(v) for v in data.values())
    print(sizes)


