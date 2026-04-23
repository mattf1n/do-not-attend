import warnings
from transformers import logging as transformers_logging
from datasets import disable_progress_bar

warnings.filterwarnings("ignore", message=".*rope_parameters.*")  # silence RoPE config type warnings
transformers_logging.set_verbosity_error()                         # suppress transformers info/warnings, errors only
disable_progress_bar()                                             # hide datasets "Resolving data files" bar

import json
import os
import time
import torch

from model import get_model, get_attentions
from tokenization import get_multi_token_words, summarize_multi_token_words
from analysis import aggregate_multi_token_word_attentions
from data import get_paragraphs, get_data_samples, PILE_COMPONENTS


def _get_max_tokens_input(default=20000):
    MAX_CONTEXT = 2 ** 16  # 65536 — model context window length
    print(f"[main] Max allowed tokens: {MAX_CONTEXT} (2^16, context window length).")
    while True:
        raw = input(f"Max tokens budget (default = {default}): ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print(f"[main] Invalid input, please enter a number.")
            continue
        if val > MAX_CONTEXT:
            print(f"[main] {val} exceeds the context window length of {MAX_CONTEXT} (2^16). Please try again.")
            continue
        return val


def _get_max_num_subtokens_input(default=2):
    raw = input(f"What cap do you want for subtokens per word (default = {default}): ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[main] Invalid input, using default={default}.")
        return default


def _run_pipeline(text, component, num_tokens, max_num_subtokens, model, tokenizer):
    """Run the full attention + multi-token-word pipeline for one (text, component) pair."""
    print(f"[main] Getting attentions from model for component='{component}'...")
    attentions = get_attentions(text, model, tokenizer)
    print("[main] Attentions extracted.")

    print("[main] Getting multi-token words from text...")
    multi_token_words_map = get_multi_token_words(text, tokenizer, max_num_subtokens)
    print(f"[main] Found {len(multi_token_words_map)} unique multi-token words.")

    print("[main] Aggregating attentions for multi-token words...")
    multi_token_word_attention_map = aggregate_multi_token_word_attentions(attentions, multi_token_words_map)
    print("[main] Aggregation complete.")

    num_unique_mt_words, num_mt_word_occurrences = summarize_multi_token_words(multi_token_words_map)  # mt = multi-token

    out = {
        'text': text,
        'component': component,
        'num_tokens': num_tokens,
        'multi_token_words_list': list(multi_token_words_map.keys()),
        'num_unique_mt_words': num_unique_mt_words,         # mt = multi-token
        'num_mt_word_occurrences': num_mt_word_occurrences, # mt = multi-token
        'main_data': multi_token_word_attention_map,
    }
    return out


def _safedump(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return obj


def _component_to_filename(component):
    return component.replace(" ", "_").replace("(", "").replace(")", "")


def _write_stats_doc(out_dir, stats_rows, max_tokens, max_num_subtokens, mode, timestamp,
                     components_used, scope_was_all, skipped, total_duration_s):
    """
    Write a human-readable stats summary markdown file to out_dir/stats.md.

    Args:
        out_dir: directory to write stats.md into
        stats_rows: list of dicts, one per successful component run, each with keys:
                    component, num_samples, actual_tokens, num_unique_mt_words,
                    num_mt_word_occurrences, duration_s
        max_tokens: the token budget used for the run
        max_num_subtokens: subtoken cap used
        mode: string describing the run mode (e.g. "single run", "multi-component (all)", "multi-component (subset)")
        timestamp: datetime string for when the run started
        components_used: list of component names that were attempted
        scope_was_all: bool, True if user selected "all" in multi-component mode
        skipped: list of component names skipped due to 0 samples
        total_duration_s: total wall-clock seconds for the run
    """
    lines = []
    lines.append("# Run Stats\n")
    lines.append(f"**Timestamp:** {timestamp}  ")
    lines.append(f"**Max tokens budget:** {max_tokens}  ")
    lines.append(f"**Max subtokens per word:** {max_num_subtokens}  ")
    lines.append(f"**Mode:** {mode}  ")
    lines.append("")

    lines.append("## Components Used")
    for c in components_used:
        lines.append(f"- {c}")
    if scope_was_all:
        lines.append("")
        lines.append("*(from \"all\" — full PILE_COMPONENTS list)*")
    lines.append("")

    if skipped:
        lines.append("## Skipped Components")
        for c in skipped:
            lines.append(f"- {c}")
        lines.append("")

    lines.append("## Per-Component Results")
    lines.append("")
    header = "| Component | num_samples | actual_tokens | num_unique_mt_words | num_mt_word_occurrences | duration (s) |"
    sep    = "|-----------|-------------|---------------|---------------------|-------------------------|--------------|"
    lines.append(header)
    lines.append(sep)
    for row in stats_rows:
        lines.append(
            f"| {row['component']} "
            f"| {row['num_samples']} "
            f"| {row['actual_tokens']} "
            f"| {row['num_unique_mt_words']} "
            f"| {row['num_mt_word_occurrences']} "
            f"| {row['duration_s']:.2f} |"
        )
    lines.append("")
    lines.append(f"**Total run duration:** {total_duration_s:.2f}s")

    stats_path = os.path.join(out_dir, "stats.md")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[main] Stats doc written to {stats_path}")


def single_run():
    print("[main] --- Single Run ---")
    run_start = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    sub_choice = input("1: paragraphs\n2: one doc\nelse: one paragraph\n> ").strip()

    if sub_choice == "1":
        num_paragraphs = int(input("How many paragraphs: "))
        print(f"[main] Fetching {num_paragraphs} paragraph(s)...")
        text = get_paragraphs(num_paragraphs)
    elif sub_choice == "2":
        from data import get_data_sample
        sample_idx_input = input("Which sample index? (default = 0): ").strip()
        sample_idx = int(sample_idx_input) if sample_idx_input.isdigit() else 0
        print(f"[main] Fetching sample index {sample_idx}...")
        text = get_data_sample(sample_idx)
    else:
        print("[main] Fetching first paragraph...")
        text = get_paragraphs()

    component_input = input("Pile component (leave blank for 'all'): ").strip()
    component = component_input if component_input else "all"

    num_tokens = _get_max_tokens_input()
    max_num_subtokens = _get_max_num_subtokens_input()

    print("[main] Sample of text input:\n", text[:300] + ("..." if len(text) > 300 else ""))

    print("[main] Loading model and tokenizer...")
    model, tokenizer = get_model()
    print("[main] Model and tokenizer loaded.")

    out = _run_pipeline(text, component, num_tokens, max_num_subtokens, model, tokenizer)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[main] Freed CUDA memory.")

    os.makedirs("output", exist_ok=True)
    component_slug = _component_to_filename(component)
    out_path = f"output/{component_slug}_{num_tokens}tokens.json"

    print(f"[main] Writing results to {out_path} ...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_safedump)
    print("[main] Output saved successfully.")

    total_duration = time.time() - run_start
    stats_row = {
        'component': component,
        'num_samples': 1,
        'actual_tokens': num_tokens,
        'num_unique_mt_words': out['num_unique_mt_words'],
        'num_mt_word_occurrences': out['num_mt_word_occurrences'],
        'duration_s': total_duration,
    }
    _write_stats_doc(
        out_dir="output",
        stats_rows=[stats_row],
        max_tokens=num_tokens,
        max_num_subtokens=max_num_subtokens,
        mode="single run",
        timestamp=timestamp,
        components_used=[component],
        scope_was_all=(component == "all"),
        skipped=[],
        total_duration_s=total_duration,
    )


def multi_component_run():
    print("[main] --- Multi-Component Run ---")
    total_start = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    num_tokens = _get_max_tokens_input()
    max_num_subtokens = _get_max_num_subtokens_input()

    scope = input("1: all components\n2: subset\nelse (default): Pile-CC\n> ").strip()

    if scope == "1":
        components = PILE_COMPONENTS
        scope_was_all = True
        mode = "multi-component (all)"
    elif scope == "2":
        print("\nSelect components (enter numbers separated by spaces):")
        for i, name in enumerate(PILE_COMPONENTS, start=1):
            print(f"  {i}: {name}")
        raw = input("\nYour selection: ").strip()
        indices = [int(x) - 1 for x in raw.split() if x.isdigit()]
        components = [PILE_COMPONENTS[i] for i in indices if 0 <= i < len(PILE_COMPONENTS)]
        if not components:
            print("[main] No valid components selected, aborting.")
            return
        scope_was_all = False
        mode = "multi-component (subset)"
    else:
        components = ["Pile-CC"]
        scope_was_all = False
        mode = "multi-component (Pile-CC default)"
        print("[main] Defaulting to Pile-CC.")

    print(f"[main] Will run for components: {components}")

    out_dir = f"output/{num_tokens}_tokens"
    os.makedirs(out_dir, exist_ok=True)

    print("[main] Loading model and tokenizer (once for all components)...")
    model, tokenizer = get_model()
    print("[main] Model and tokenizer loaded.")

    stats_rows = []
    skipped = []

    for component in components:
        comp_start = time.time()
        print(f"\n[main] ===== Processing component: '{component}' =====")
        print(f"[main] Fetching up to {num_tokens} tokens of data...")
        text, metadata = get_data_samples(
            component=component,
            max_tokens=num_tokens,
            type="string",
            return_metadata=True,
        )
        # Guard: skip if no samples were collected (empty component or first sample exceeds token budget)
        if metadata["num_samples"] == 0:
            print(f"[main] Skipping '{component}': 0 samples collected (component may be empty or first sample exceeds token budget of {num_tokens}).")
            skipped.append(component)
            continue
        print(f"[main] Data fetched: {metadata}")
        print("[main] Sample of text:\n", text[:300] + ("..." if len(text) > 300 else ""))

        out = _run_pipeline(text, component, num_tokens, max_num_subtokens, model, tokenizer)

        component_slug = _component_to_filename(component)
        out_path = f"{out_dir}/{component_slug}_{num_tokens}tokens.json"

        print(f"[main] Writing results to {out_path} ...")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=_safedump)
        print(f"[main] Saved: {out_path}")

        stats_rows.append({
            'component': component,
            'num_samples': metadata['num_samples'],
            'actual_tokens': metadata['num_tokens'],
            'num_unique_mt_words': out['num_unique_mt_words'],
            'num_mt_word_occurrences': out['num_mt_word_occurrences'],
            'duration_s': time.time() - comp_start,
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[main] Freed CUDA memory.")

    total_duration = time.time() - total_start
    _write_stats_doc(
        out_dir=out_dir,
        stats_rows=stats_rows,
        max_tokens=num_tokens,
        max_num_subtokens=max_num_subtokens,
        mode=mode,
        timestamp=timestamp,
        components_used=components,
        scope_was_all=scope_was_all,
        skipped=skipped,
        total_duration_s=total_duration,
    )

    print("\n[main] All components complete.")


def main():
    print("[main] Starting...")
    mode = input("1: single run\n2: multi-component run\n> ").strip()
    if mode == "2":
        multi_component_run()
    else:
        single_run()


if __name__ == "__main__":
    main()
