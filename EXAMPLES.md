# Examples

## Check the source text of a JSON file

```python
from utils import load_json

data = load_json('output/16000_tokens/USPTO_Backgrounds_16000tokens.json')
print(data['text'])
```

## Load bywords and apply filters

```python
from testing import get_bywords, apply_filters
path = 'output/16000_tokens/Wikipedia_en_16000tokens.json'
bywords = get_bywords(path)
print(f'Loaded {len(bywords)} bywords:\n{bywords}\n')
apply_filters(bywords)
```
