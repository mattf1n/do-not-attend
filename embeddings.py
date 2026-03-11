import re
import string
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import json
from collections import Counter
import seaborn


from utils import *
from data import get_paragraphs
from model import get_model
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


text = get_paragraphs()

model, tokenizer = get_model()
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    #use_cachemodel(**inputs, use_cache=True, return_dict = =True: return keys and values
    outputs = True)
past_key_values = outputs.past_key_values
# This is a tuple of (key, value) per layer
# Each key tensor has shape: (batch, num_kv_heads, seq_len, head_dim)

#To get all key embeddings for a specific token position across all layers:

token_pos = 5  # position of your token in the sequence
# Key embeddings for that token at every layer

#shape of .keys: (batch, num_kv_heads, seq_len, head_dim). 
# batch = 1 (only one input string being used here)
all_keys = [past_key_values.layers[layer].keys[:, :, token_pos, :]
          for layer in range(len(past_key_values))]
# all_keys[i] has shape (1, num_kv_heads, head_dim)

If you want to find the position of a specific token first:

token_ids = inputs["input_ids"][0]
target_id = tokenizer.encode("the", add_special_tokens=False)[0]
positions = (token_ids == target_id).nonzero(as_tuple=True)[0]
