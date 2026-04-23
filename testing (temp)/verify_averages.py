"""
Verify that the new averaged output format matches the old full-row format.

New format: heads[head_idx] = [mean_att_subtoken0, mean_att_subtoken1]
  where each value is the mean of all attending rows (rows from start: onward)
  for that subtoken column.

Old format: heads[head_idx][subtoken_idx] = full vector of attending-row values
  (one scalar per attending row, from position start = max(token_indices)+1 onward).

Verification: np.mean(old_heads[h][k]) should equal new_heads[h][k] within tolerance.

Note on tolerance: values are float32, serialised to JSON then parsed back as float64.
np.mean over a long float64 array accumulates differently than PyTorch's float32 mean,
so a tolerance of 1e-4 is appropriate (largest observed rounding diff: ~5e-5).
"""

import json
import sys
import numpy as np

NEW_PATH = "output/10_paragraphs_2-token_max_word_output_sample0.json"
OLD_PATH = "output/all_att/10_paragraphs_2-token_max_word_output.json"
TOLERANCE = 1e-4 # NOT 0 b/c of float32 rounding artifacts
MAX_MISMATCHES_TO_PRINT = 20


def load(path):
    print(f"Loading {path} ...")
    with open(path) as f:
        return json.load(f)


def verify(new_data, old_data, tol=TOLERANCE):
    new_main = new_data["main_data"]
    old_main = old_data["main_data"]

    total = 0
    mismatches = []

    words_only_in_new = set(new_main) - set(old_main)
    words_only_in_old = set(old_main) - set(new_main)
    if words_only_in_new:
        print(f"WARNING: {len(words_only_in_new)} word(s) in new but not old: {words_only_in_new}")
    if words_only_in_old:
        print(f"WARNING: {len(words_only_in_old)} word(s) in old but not new: {words_only_in_old}")

    for word in new_main:
        if word not in old_main:
            continue

        new_occs = new_main[word]["occurrences"]
        old_occs = old_main[word]["occurrences"]

        if len(new_occs) != len(old_occs):
            print(f"WARNING: '{word}' has {len(new_occs)} occurrences in new but {len(old_occs)} in old")

        for occ_idx, (new_occ, old_occ) in enumerate(zip(new_occs, old_occs)):
            token_indices = new_occ["token_indices"]

            new_layers = new_occ["attentions"]["layers"]
            old_layers = old_occ["attentions"]["layers"]

            if len(new_layers) != len(old_layers):
                print(f"WARNING: '{word}' occ {occ_idx} layer count mismatch: {len(new_layers)} vs {len(old_layers)}")

            for layer_idx, (new_layer, old_layer) in enumerate(zip(new_layers, old_layers)):
                new_heads = new_layer["heads"]
                old_heads = old_layer["heads"]

                if len(new_heads) != len(old_heads):
                    print(f"WARNING: '{word}' occ {occ_idx} layer {layer_idx} head count mismatch")

                for head_idx, (new_head, old_head) in enumerate(zip(new_heads, old_heads)):
                    # new_head: list of K scalars (one per subtoken)
                    # old_head: list of K lists (one list per subtoken, each list = attending rows)
                    for k, (new_val, old_rows) in enumerate(zip(new_head, old_head)):
                        expected = float(np.mean(old_rows))
                        actual = float(new_val)
                        total += 1

                        if abs(expected - actual) > tol:
                            mismatches.append({
                                "word": word,
                                "occ_idx": occ_idx,
                                "token_indices": token_indices,
                                "layer": layer_idx,
                                "head": head_idx,
                                "subtoken": k,
                                "expected": expected,
                                "actual": actual,
                                "diff": abs(expected - actual),
                            })

    return total, mismatches


def main():
    new_data = load(NEW_PATH)
    old_data = load(OLD_PATH)

    print(f"Verifying with tolerance = {TOLERANCE}\n")
    total, mismatches = verify(new_data, old_data)

    print(f"Total checks : {total:,}")
    print(f"Mismatches   : {len(mismatches):,}")

    if not mismatches:
        print("\nPASS — all averages match.")
        return

    print(f"\nFAIL — showing first {min(MAX_MISMATCHES_TO_PRINT, len(mismatches))} mismatches:\n")
    for m in mismatches[:MAX_MISMATCHES_TO_PRINT]:
        print(
            f"  word={m['word']!r:20s}  occ={m['occ_idx']}  tokens={m['token_indices']}"
            f"  L{m['layer']:02d} H{m['head']:02d} sub{m['subtoken']}"
            f"  expected={m['expected']:.8f}  actual={m['actual']:.8f}  diff={m['diff']:.2e}"
        )

    if len(mismatches) > MAX_MISMATCHES_TO_PRINT:
        print(f"  ... and {len(mismatches) - MAX_MISMATCHES_TO_PRINT} more.")

    max_diff = max(m["diff"] for m in mismatches)
    print(f"\nLargest difference: {max_diff:.2e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
