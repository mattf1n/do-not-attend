# Do Not Attend - Project Context for AI Assistants

## Project Overview

This project investigates attention mechanisms in large language models (LLMs), specifically examining whether attention patterns differ between subtokens of multi-token words. The core hypothesis is that for a multi-token word like `[mul][ti][ple]`, the model should attend more strongly to the **last** subtoken (`ple`) than to earlier subtokens (`mul`, `ti`) when processing subsequent tokens.

### Research Question
For multi-token words tokenized as [A][B], do later tokens in the sequence preferentially attend to subtoken B over subtoken A?

### Model & Dataset
- **Model**: OLMo-3-7B (context window: 65,536 tokens / 2^16)
- **Dataset**: Pile-uncopyrighted corpus (various components like Pile-CC, Wikipedia, PubMed, etc.)
- **Dataset source**: [monology/pile-uncopyrighted](https://huggingface.co/datasets/monology/pile-uncopyrighted)

## Core Concepts

### Multi-Token Words
Words that are split into multiple tokens by the tokenizer. For example:
- `"2000"` might tokenize as `["20", "00"]`
- `"California"` might tokenize as `["Cal", "ifor", "nia"]`

The project tracks these words and analyzes attention patterns for each subtoken.

### Attention Aggregation
For each multi-token word occurrence at positions `[i, j]`, the code:
1. Extracts attention weights from all rows after position `j` (attending rows)
2. Computes mean attention to each subtoken across all attending rows
3. Stores these per-occurrence, per-head averages for analysis

### Key Metrics

1. **Hypothesis Rate**: Fraction of occurrences where `mean_attention(last_subtoken) > mean_attention(first_subtoken)`

2. **Michelson Contrast**: Scale-invariant measure of attention difference:
   ```
   contrast = (s1 - s0) / (s1 + s0)
   ```
   where s0/s1 are mean attentions to first/last subtokens. Range: [-1, 1]

3. **Micro-averaging**: Pool all occurrences, compute metrics (weighted by frequency)

4. **Macro-averaging**: Average per word type first, then across words (equal weight per word)

## Project Structure

### Core Pipeline Files

- **`main.py`**: Entry point with two modes:
  - **Attention run**: Extracts attention weights, aggregates by multi-token word
  - **QKV slot run**: Extracts Q/K/V vectors per subtoken for geometric analysis
  
- **`run_experiments.py`**: Runs hypothesis tests on saved data
  - Exp 2: Box-whisker plots per head
  - Exp 3: Vector polar coordinates (k0 vs k1)
  - Exp 4: Hypothesis rate analysis (heatmaps + bar charts)
  - Exp 5: Michelson contrast analysis
  - Exp 6-7: Pooled versions (multi-component aggregation)
  - Exp 8: All QKV slot-pair polar heatmaps

- **`analysis.py`**: Core analysis functions
  - `aggregate_multi_token_word_attentions`: Main aggregation logic
  - Streaming aggregators: `MultiTokenWordAggregator`, `HeadByHeadAggregator`
  - Hypothesis rate and contrast computation functions
  - Pooled (micro) and macro-averaged variants
  - Word filtering by category

- **`qkv_vectors.py`**: Q/K/V vector extraction and geometric analysis
  - Extracts post-RoPE vectors from TransformerLens cache
  - Polar coordinate analysis between subtoken pairs
  - Slot file format: one `.pt` per role+index (q0, q1, k0, k1, v0, v1)

### Support Files

- **`model.py`**: Model loading (HuggingFace + TransformerLens bridge)
- **`tokenization.py`**: Multi-token word identification
- **`data.py`**: Pile dataset loading and sampling
- **`visualizations.py`**: All plotting functions (heatmaps, bar charts, polar grids)
- **`utils.py`**: JSON/NPZ I/O, word classification (numbers, space_numbers, words, etc.)
- **`config.py`**: Model/dataset configuration

### Scripts

- **`scripts/run_qkv_cache.sh`**: SBATCH job for QKV extraction on HPC
- **`scripts/run_experiments.sh`**: SBATCH job for analysis
- **`scripts/run_job_discovery.sh`** & **`run_job_endeavour.sh`**: Cluster-specific launchers

## Running Experiments

### Interactive Mode (Attention Run)

```bash
python main.py
# Select mode 1 (attention) or 2 (QKV)
# Choose components: all, subset, or default (Pile-CC)
# Set token budget (default: 20000, max: 65536)
# Set subtoken cap per word (default: 2)
```

**Output**: `output/{num_tokens}_tokens/{component}_{num_tokens}tokens.json`

### Batch Mode (Non-interactive)

```bash
# Attention run
python main.py --batch --tokens 16000 --max-subtokens 2 --components "Pile-CC,Wikipedia" --mode attention

# QKV slot extraction
python main.py --batch --tokens 16000 --max-subtokens 2 --components all --mode qkv
```

### Running Analysis

```bash
# Single JSON file
python run_experiments.py output/my_output.json

# Specific experiments only
python run_experiments.py output/my_output.json --exp 4 5

# Multi-component folder (runs exp 6-7 pooled analysis)
python run_experiments.py --folder output/16000_tokens/ --exp 4 5 6 7

# QKV polar heatmaps (requires slot directory)
python run_experiments.py --exp 8 --pt output/qkv_cache/16000_tokens/Pile-CC_16000tokens/

# Filter by word category
python run_experiments.py output/my_output.json --exp 4 5 --filter numbers

# Run all filters
python run_experiments.py --folder output/16000_tokens/ --exp 4 5 --all-filters
```

**Output**: `figures/{token_folder}/{component}/attention/{label}/`

### Word Categories (Filters)

Defined in `utils.py`:
- `newlines`: Words containing `\n`
- `space_numbers`: Start with space + digit (e.g., `" 2000"`)
- `space_words`: Start with space + letter
- `space_symbols`: Start with space + symbol
- `numbers`: Pure digit strings
- `words`: Pure letter strings
- `symbols`: Pure symbol strings
- `other`: Everything else

## Key Implementation Details

### Memory Optimization

The project handles large contexts (up to 65K tokens) through several optimization strategies:

1. **Layer-at-a-time processing** (`MultiTokenWordAggregator`):
   - Processes attention weights one layer at a time
   - Reduces peak memory from L×(H×S×S) to 1×(H×S×S)
   - Enables ~45K token contexts on 150GB RAM (vs ~7K without)

2. **Head-by-head streaming** (`HeadByHeadAggregator`):
   - Processes one attention head at a time
   - 32× further reduction for 32-head models
   - Slower but enables even larger contexts

3. **Forward hooks**: Intercepts attention weights during forward pass, aggregates immediately, discards raw tensors

### Data Schema

**Multi-token word map** (used throughout pipeline):
```json
{
  "word_string": {
    "occurrences": [
      {
        "token_indices": [42, 43],
        "attentions": {
          "layers": [
            {"heads": [tensor_head0, tensor_head1, ...]},
            ...
          ]
        }
      }
    ]
  }
}
```

Each `tensor_head` has shape `[num_subtokens]` containing mean attention scores.

**QKV slot files** (`.pt` format):
```python
{
  "word_string": {
    (layer_idx, head_idx): [
      torch.Tensor(...),  # occurrence 1 vector
      torch.Tensor(...),  # occurrence 2 vector
      ...
    ]
  }
}
```

### Output Formats

1. **JSON**: Human-readable, includes full text and metadata (larger files)
2. **NPZ**: Binary format via `save_output_npz()` in `utils.py` (smaller, faster)
   - `run_experiments.py --npz` transparently converts to temp JSON

## Common Operations

### Inspecting Word Frequencies

```python
from analysis import rank_words_by_occurrence
ranked = rank_words_by_occurrence("output/my_output.json", top_n=20)
print(ranked)  # [(word, count), ...]
```

### Generating Filter Stats

```python
from analysis import generate_filter_stats
stats = generate_filter_stats("output/my_output.json")
print(stats)
```

### Converting JSON to NPZ

```python
from utils import save_output_npz
save_output_npz("output/my_output.json", "output/binary/")
# Creates: output/binary/my_output/my_output.npz + my_output_meta.json
```

## References

Key papers informing this work:

- **Feucht et al. (2024)**: Token erasure as implicit vocabulary items (EMNLP)
- **Kallini et al. (2025)**: MrT5 dynamic token merging (ICLR)
- **Kamoda et al. (2025)**: Weight-based detokenization analysis (NAACL Findings)
- **Lad et al. (2025)**: Robustness and inference stages in LLMs
- **Liu et al. (2025)**: SuperBPE space travel for language models
- **Park et al. (2025)**: Geometry of categorical concepts in LLMs (ICLR)

## Important Notes

1. **Concatenation with `\n\n`**: Multiple samples are joined with double newlines to create single strings for model input

2. **No BOS token**: TransformerLens runs use `prepend_bos=False` to match HuggingFace behavior

3. **Post-RoPE vectors**: QKV analysis uses post-RoPE embeddings (`hook_rot_q`, `hook_rot_k`, `v.hook_out`)

4. **Overwrite behavior**: Scripts prompt before overwriting existing output directories or auto-create numbered variants `(1)`, `(2)`, etc.

5. **Stats docs**: Multi-component runs generate `stats.md` with per-component metadata tables

6. **Pooled experiments**: Only available in `--folder` mode (exp 6-7), micro-average across components

## Development Tips

- **Testing small contexts**: Use `--tokens 500` for quick iteration
- **GPU memory**: Attention runs work on CPU; QKV extraction benefits from GPU
- **Debugging aggregation**: See `compare_streaming.py` and `testing.py` for validation scripts
- **Custom word filters**: Add categories to `WORD_CATEGORIES` in `utils.py` and update `classify_word()`
- **Plot customization**: All visualization functions in `visualizations.py` accept context strings for titles
