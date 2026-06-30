from data import get_data_samples

# from data import PILE_COMPONENTS
from tokenization import get_multi_token_words, summarize_multi_token_words
from model import get_tokenizer

import warnings
import json
import torch
from transformers import logging as transformers_logging
from datasets import disable_progress_bar

# Silence specific warnings and unnecessary output
warnings.filterwarnings("ignore", message=".*rope_parameters.*")
transformers_logging.set_verbosity_error()
disable_progress_bar()


from data import PILE_COMPONENTS
# PILE_COMPONENTS = [
#     "Pile-CC",
#     "PubMed Central",
#     # "ArXiv",
#     # "GitHub",
#     "FreeLaw",
#     # "Stack Exchange",
#     "USPTO Backgrounds",
#     # "DM Mathematics",
#     "Wikipedia (en)",
#     "HackerNews",
#     "NIH ExPorter",
#     "PubMed Abstracts",
#     "Enron Emails",
#     # "EuroParl",
#     "PhilPapers",
#     # "Ubuntu IRC",
#     "Gutenberg (PG-19)",
# ]


def analyze_single_component():
    tokenizer = get_tokenizer()
    print("Available dataset components:")
    for idx, comp in enumerate(PILE_COMPONENTS, 1):
        print(f"{idx}. {comp}")
    try:
        comp_choice = int(input("Enter the number of the dataset component to analyze: "))
        component = PILE_COMPONENTS[comp_choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return
    max_tokens = int(input(f"Max tokens for component '{component}': "))
    print(f"\n=== {component} ===")
    text = get_data_samples(component=component, max_tokens=max_tokens, type="string")
    multi_token_map = get_multi_token_words(text, tokenizer)
    summarize_multi_token_words(multi_token_map)

def analyze_all_components():
    tokenizer = get_tokenizer()
    max_tokens = int(input("Max tokens per dataset component: "))
    for component in PILE_COMPONENTS:
        print(f"\n=== {component} ===")
        text = get_data_samples(component=component, max_tokens=max_tokens, type="string")
        multi_token_map = get_multi_token_words(text, tokenizer)
        summarize_multi_token_words(multi_token_map)

def rank_occurrences_cli():
    import os
    import json

    output_root = os.path.join(os.path.dirname(__file__), "output")

    folders = sorted(
        f for f in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, f))
    )
    print("\nAvailable output folders:")
    for idx, name in enumerate(folders, 1):
        print(f"  {idx}. {name}")
    try:
        folder_choice = int(input("Select folder number: "))
        folder = folders[folder_choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    folder_path = os.path.join(output_root, folder)
    json_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".json"))
    if not json_files:
        print(f"No JSON files found in output/{folder}/")
        return

    components = []
    for fname in json_files:
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath) as f:
                comp = json.load(f).get("component", fname)
            components.append((comp, fpath))
        except Exception:
            pass

    print(f"\nAvailable components in '{folder}':")
    for idx, (comp, _) in enumerate(components, 1):
        print(f"  {idx}. {comp}")
    try:
        comp_choice = int(input("Select component number: "))
        comp_name, comp_path = components[comp_choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    top_n_raw = input("How many top words to show? (leave blank for all): ").strip()
    top_n = int(top_n_raw) if top_n_raw else None

    from analysis import rank_words_by_occurrence
    ranked = rank_words_by_occurrence(comp_path, top_n=top_n)

    print(f"\nWord occurrence ranking for '{comp_name}' ({folder}):")
    print(f"  {'Rank':<6} {'Word':<35} {'Occurrences':>12}")
    print("  " + "-" * 55)
    for rank, (word, count) in enumerate(ranked, 1):
        print(f"  {rank:<6} {repr(word):<35} {count:>12}")


if __name__ == "__main__":
    mode = input(
        "Choose mode:\n"
        "  1. Analyze one component (tokenization)\n"
        "  2. Analyze all components (tokenization)\n"
        "  3. Rank words by occurrence from output JSON\n"
        "Enter 1, 2, or 3: "
    ).strip()
    if mode == "1":
        analyze_single_component()
    elif mode == "2":
        analyze_all_components()
    elif mode == "3":
        rank_occurrences_cli()
    else:
        print("Invalid input.")