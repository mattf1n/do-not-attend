import json
import torch

from model import get_model, get_attentions
from tokenization import get_multi_token_words, get_multi_token_words_ai
from analysis import aggregate_multi_token_word_attentions, aggregate_multi_token_word_attentions_ai
from data import get_paragraphs

def main():
   
   test_multitoken()



def test_multitoken(output_json_path="multi_word_output_1.json"):
    print("[test_multitoken] Starting test pipeline...")

    choice = input("1: paragraph \n2: one doc \nelse: one paragraph ")
    print(f"[test_multitoken] User input choice: {choice}")

    if (choice == "1"):
        print("[test_multitoken] Paragraph selected")
        num_paragraphs = int(input("How many paragraphs: "))
        print(f"[test_multitoken] Number of paragraphs requested: {num_paragraphs}")
        text = get_paragraphs(num_paragraphs)
    elif (choice == "2"):
        print("[test_multitoken] One doc selected")
        text = get_data_sample()
    else:
        print("[test_multitoken] Default selected (one paragraph)")
        text = get_first_paragraph()

    print("[test_multitoken] Sample of text input:\n", text[:300] + ("..." if len(text) > 300 else ""))
    
    num_tokens = input("What cap do you want for tokens (default = 2): ")
    if not num_tokens:
        num_tokens = 2
    else:
        try:
            num_tokens = int(num_tokens)
        except Exception as e:
            print(f"[test_multitoken] Warning: invalid input for num_tokens. Using default=2. [error: {e}]")
            num_tokens = 2
    print(f"[test_multitoken] Cap of {num_tokens} selected")

    print("[test_multitoken] Loading model and tokenizer...")
    model, tokenizer = get_model()
    print("[test_multitoken] Model and tokenizer loaded.")

    print("[test_multitoken] Getting attentions from model...")
    attentions = get_attentions(text, model, tokenizer)
    print("[test_multitoken] Attentions extracted.")

    # Free GPU memory from the model before heavy analysis
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[test_multitoken] Freed CUDA memory.")

    print("[test_multitoken] Getting multi-token words from text...")
    multi_token_words_map = get_multi_token_words(text, tokenizer, num_tokens)
    print(f"[test_multitoken] Found {len(multi_token_words_map)} multi-token words.")

    print("[test_multitoken] Aggregating attentions for multi-token words...")
    multi_token_word_attention_map = aggregate_multi_token_word_attentions(attentions, multi_token_words_map)
    print("[test_multitoken] Aggregation complete.")

    out = multi_token_word_attention_map

    # Convert everything to serializable form for JSON
    # Safe serialization: convert any numpy types or non-serializable elements
    def safedump(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    print(f"[test_multitoken] Writing results to output file: {output_json_path} ...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=safedump)
    
    print("[test_multitoken] Output saved successfully.")


def test_multitoken_ai(output_json_path="output/multi_word_output_ai.json"):
    print("[test_multitoken_ai] Starting test pipeline...")

    choice = input("1: paragraph \n2: one doc \nelse: one paragraph ")
    print(f"[test_multitoken_ai] User input choice: {choice}")

    if (choice == "1"):
        print("[test_multitoken_ai] Paragraph selected")
        num_paragraphs = int(input("How many paragraphs: "))
        print(f"[test_multitoken_ai] Number of paragraphs requested: {num_paragraphs}")
        text = get_paragraphs(num_paragraphs)
    elif (choice == "2"):
        print("[test_multitoken_ai] One doc selected")
        text = get_data_sample()
    else:
        print("[test_multitoken_ai] Default selected (one paragraph)")
        text = get_first_paragraph()

    print("[test_multitoken_ai] Sample of text input:\n", text[:300] + ("..." if len(text) > 300 else ""))

    num_tokens = input("What cap do you want for tokens (default = 2): ")
    if not num_tokens:
        num_tokens = 2
    else:
        try:
            num_tokens = int(num_tokens)
        except Exception as e:
            print(f"[test_multitoken_ai] Warning: invalid input for num_tokens. Using default=2. [error: {e}]")
            num_tokens = 2
    print(f"[test_multitoken_ai] Cap of {num_tokens} selected")

    print("[test_multitoken_ai] Loading model and tokenizer...")
    model, tokenizer = get_model()
    print("[test_multitoken_ai] Model and tokenizer loaded.")

    print("[test_multitoken_ai] Getting attentions from model...")
    attentions = get_attentions(text, model, tokenizer)
    print("[test_multitoken_ai] Attentions extracted.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[test_multitoken_ai] Freed CUDA memory.")

    print("[test_multitoken_ai] Getting multi-token words from text (AI-friendly schema)...")
    multi_token_words_map = get_multi_token_words_ai(text, tokenizer, num_tokens)
    print(f"[test_multitoken_ai] Found {len(multi_token_words_map)} multi-token words.")

    print("[test_multitoken_ai] Aggregating attentions for multi-token words (AI-friendly schema)...")
    out = aggregate_multi_token_word_attentions_ai(attentions, multi_token_words_map)
    print("[test_multitoken_ai] Aggregation complete.")

    def safedump(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    print(f"[test_multitoken_ai] Writing results to output file: {output_json_path} ...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=safedump)

    print("[test_multitoken_ai] Output saved successfully.")

if __name__ == "__main__":  

    # pick = input("1: Visualization \nelse: pipeline")

    # if pick == "1": 
    #     # plot_by_word_length_hist_summary()
    # else: 
    test_multitoken_ai()
    #     text, multi, results, max_summary = test_pipeline()
    #     pprint(max_summary)

