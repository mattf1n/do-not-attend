import json
import torch

from model import get_model, get_attentions
from tokenization import get_multi_token_words
from analysis import aggregate_multi_token_word_attentions
from data import get_paragraphs


def test_multitoken():
    print("[test_multitoken] Starting test pipeline...")

    choice = input("1: paragraph \n2: one doc \nelse: one paragraph ")
    print(f"[test_multitoken] User input choice: {choice}")

    num_paragraphs = None  # Track this if applicable
    sample_idx = 0

    if choice == "1":
        actual_choice = "paragraph"
        print("[test_multitoken] Paragraph selected")
        num_paragraphs = int(input("How many paragraphs: "))
        print(f"[test_multitoken] Number of paragraphs requested: {num_paragraphs}")
        text = get_paragraphs(num_paragraphs)
    elif choice == "2":
        actual_choice = "one doc"
        print("[test_multitoken] One doc selected")
        from data import get_data_sample
        sample_idx_input = input("Which sample index? (default = 0): ")
        sample_idx = int(sample_idx_input) if sample_idx_input.strip().isdigit() else 0
        print(f"[test_multitoken] Using sample index: {sample_idx}")
        text = get_data_sample(sample_idx)
        num_paragraphs = len(text.split('\n\n'))
    else:
        actual_choice = "one paragraph"
        print("[test_multitoken] Default selected (one paragraph)")
        num_paragraphs = 1
        text = get_paragraphs()  # first paragraph


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

    out = { 
        'text': text,
        'choice': actual_choice,
        'num_paragraphs': num_paragraphs,
        'sample_index': sample_idx,
        'attention_type': "avg" # for each occurence we take avg value by row for each bitoken in each head
        'main_data': multi_token_word_attention_map,
        
    }

    # Convert everything to serializable form for JSON
    # Safe serialization: convert any numpy types or non-serializable elements
    def safedump(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    # More robust, clear, and typo-proof output filename generation:
    out_path = f"output/{num_paragraphs}_paragraphs_{num_tokens}-token_max_word_output_sample{sample_idx}.json"

    print(f"[test_multitoken] Writing results to output file: {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=safedump)
    
    print("[test_multitoken] Output saved successfully.")

if __name__ == "__main__":  
    test_multitoken()
