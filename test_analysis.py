import torch
import timeit
from analysis import aggregate_multi_token_word_attentions, aggregate_multi_token_word_attentions_ai

torch.manual_seed(42)
L, H, S = 4, 6, 30

attentions = tuple(torch.randn(1, H, S, S) for _ in range(L))

multi_word_map = {
    "running":  {"occurrences": [{"token_indices": [3, 4]},   {"token_indices": [15, 16]}]},
    "unbelievable": {"occurrences": [{"token_indices": [7, 8, 9]}]},
    "it":       {"occurrences": [{"token_indices": [1]}]},     # single-subtoken edge case
}

N = 500
t_og  = timeit.timeit(lambda: aggregate_multi_token_word_attentions(attentions, multi_word_map), number=N)
t_new = timeit.timeit(lambda: aggregate_multi_token_word_attentions_ai(attentions, multi_word_map), number=N)
print(f"OG:  {t_og/N*1000:.3f} ms/call")
print(f"New: {t_new/N*1000:.3f} ms/call")
print(f"Speedup: {t_og/t_new:.2f}x")

og  = aggregate_multi_token_word_attentions(attentions, multi_word_map)
new = aggregate_multi_token_word_attentions_ai(attentions, multi_word_map)

all_match = True
for word in multi_word_map:
    for i, (occ_og, occ_new) in enumerate(zip(og[word]["occurrences"], new[word]["occurrences"])):
        assert occ_og["token_indices"] == occ_new["token_indices"], \
            f"{word}[{i}] token_indices mismatch"
        for l, (layer_og, layer_new) in enumerate(
            zip(occ_og["attentions"]["layers"], occ_new["attentions"]["layers"])
        ):
            for h, (head_og, head_new) in enumerate(
                zip(layer_og["heads"], layer_new["heads"])
            ):
                match = torch.allclose(head_og, head_new)
                if not match:
                    print(f"MISMATCH: word='{word}' occ={i} layer={l} head={h}")
                    print(f"  og shape:  {head_og.shape},  new shape: {head_new.shape}")
                    print(f"  max diff:  {(head_og - head_new).abs().max()}")
                    all_match = False

if all_match:
    print("All outputs match!")
else:
    print("Some outputs did NOT match.")
