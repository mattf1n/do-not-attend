import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DEFAULT_MODEL


def get_model(model_name=DEFAULT_MODEL):
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_attentions(input_data, model, tokenizer):
    '''
    from input string, run inference and get the attention scores of the model
    '''
    inputs = tokenizer(input_data, return_tensors="pt")
    # Move inputs to DEVICE
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()} 
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    return outputs.attentions

