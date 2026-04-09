# Plan
- data: ~~dolma 3 (wiki, 10 gb)~~ pile-uncopyrighted: [monology/pile-uncopyrighted · Datasets at Hugging Face](https://huggingface.co/datasets/monology/pile-uncopyrighted)

- model: Olma 3 7B

1. ID multi-token words in a document
   `[mul][ti][ple]`
For each layer, attention head.
2. Find maximum attn score on `mul` over all tokens following `ple`.
3. Find maximum attn score on `ple` over all token following `ple`.


We expect attn on `ple` to be higher.
Is this true?

running `run_experiments.py`

All experiments from a JSON file:
```
python run_experiments.py output/my_output.json
```

All experiments from an npz folder (created by `save_output_npz` in `utils.py`):
```
python run_experiments.py --npz output/binary/my_output/
```

To run specific experiments (e.g. 1 and 4):
```
python run_experiments.py output/my_output.json --exp 1 4
python run_experiments.py --npz output/binary/my_output/ --exp 1 4
```

**Saving output as npz** (in Python, after running `main.py`):
```python
from utils import save_output_npz
save_output_npz("output/my_output.json", "output/binary/")
# creates output/binary/my_output/my_output.npz + my_output_meta.json
```

Note: when `--npz` is used, the data is loaded into memory and written to a temporary JSON file behind the scenes, which is deleted automatically when the run finishes.


# References

Feucht, Sheridan, David Atkinson, Byron C. Wallace, and David Bau. 2024.
“Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs.”
*EMNLP*, 9727–39. <https://aclanthology.org/2024.emnlp-main.543>.

Kallini, Julie, Shikhar Murty, Christopher D Manning, Christopher Potts,
and Róbert Csordás. 2025. “MrT5: Dynamic Token Merging for Efficient
Byte-Level Language Models.” *The Thirteenth International Conference on
Learning Representations*. <https://openreview.net/forum?id=VYWBMq1L7H>.

Kamoda, Go, Benjamin Heinzerling, Tatsuro Inaba, Keito Kudo, Keisuke
Sakaguchi, and Kentaro Inui. 2025. “Weight-Based Analysis of
Detokenization in Language Models: Understanding the First Stage of
Inference Without Inference.” *NAACL (Findings)*, 6324–43.
<https://aclanthology.org/2025.findings-naacl.355/>.

Lad, Vedang, Jin Hwa Lee, Wes Gurnee, and Max Tegmark. 2025. *The
Remarkable Robustness of LLMs: Stages of Inference?*
<https://arxiv.org/abs/2406.19384>.

Liu, Alisa, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A.
Smith, and Yejin Choi. 2025. “SuperBPE: Space Travel for Language
Models.” *Second Conference on Language Modeling*.
<https://openreview.net/forum?id=lcDRvffeNP>.

Park, Kiho, Yo Joong Choe, Yibo Jiang, and Victor Veitch. 2025. “The
Geometry of Categorical and Hierarchical Concepts in Large Language
Models.” *The Thirteenth International Conference on Learning
Representations*. <https://openreview.net/forum?id=bVTM2QKYuA>.
