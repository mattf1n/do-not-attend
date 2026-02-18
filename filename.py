# coding: utf-8
1 + 1
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
model_path= "allenai/Olmo-3-1025-7B"
olmo = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
get_ipython().run_line_magic('notebook', 'my_session.ipynb')
get_ipython().run_line_magic('clear', '')
data
get_ipython().run_line_magic('save', 'filename.py 1-10')
