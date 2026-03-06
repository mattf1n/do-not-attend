from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DEFAULT_MODEL


def get_model(model_name=DEFAULT_MODEL):
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
