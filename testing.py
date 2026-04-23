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

PILE_COMPONENTS = [
    "Pile-CC",
    "PubMed Central",
    # "ArXiv",
    # "GitHub",
    "FreeLaw",
    # "Stack Exchange",
    "USPTO Backgrounds",
    # "DM Mathematics",
    "Wikipedia (en)",
    "HackerNews",
    "NIH ExPorter",
    "PubMed Abstracts",
    "Enron Emails",
    # "EuroParl",
    "PhilPapers",
    # "Ubuntu IRC",
    "Gutenberg (PG-19)",
]


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

if __name__ == "__main__":
    mode = input("Analyze (1) one component or (2) all components? Enter 1 or 2: ").strip()
    if mode == "1":
        analyze_single_component()
    elif mode == "2":
        analyze_all_components()
    else:
        print("Invalid input. Please enter 1 or 2.")