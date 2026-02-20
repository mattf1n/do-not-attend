

### Example Input

**Paragraph:**

> It is done, and submitted. You can play "Survival of the Tastiest" on Android, and on the web. Playing on the web works, but you have to simulate multi-touch for table moving and that can be a bit confusing.

---

### Multi-Token Words Detected

```python
{
    'Survival':  {'tokens': ['Surv', 'ival'],        'positions': [11, 12]},
    'Tastiest':  {'tokens': ['T', 'ast', 'iest'],    'positions': [15, 16, 17]},
    'confusing': {'tokens': ['conf', 'using'],        'positions': [50, 51]},
}
```

---

### Pipeline Output Format

Each multi-token word produces a dict with these keys:

| Key              | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `tokens`         | String tokens that compose the word                                |
| `positions`      | Token positions in the sequence                                    |
| `top_k`          | Number of tokens that make up the word                             |
| `max_per_head`   | Average per-head top-k attended tokens for each token in the word  |
| `max_global`     | Top 5 most-attended tokens across all heads/layers                 |

---

### Example Output — `result['Survival']`

**`max_per_head` (layer 31, head 0):**

```python
{'layer': 31, 'head': 0, 'top_k': [(',', 0.668), ('It', 0.204)]}
```

**`max_global`:**

| Token   | Top Attended Tokens                                                              |
| ------- | -------------------------------------------------------------------------------- |
| `Surv`  | `','` 840 (82.0%) · `'It'` 50 (4.9%) · `'"'` 47 (4.6%) · `'.'` 35 (3.4%) · `'play'` 28 (2.7%) |
| `ival`  | `','` 816 (79.7%) · `'"'` 53 (5.2%) · `'It'` 41 (4.0%) · `'.'` 35 (3.4%) · `'Surv'` 31 (3.0%) |
```
