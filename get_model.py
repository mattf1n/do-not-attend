from transformers import AutoModelForCausalLM, AutoTokenizer

model_path= "allenai/Olmo-3-1025-7B"
olmo = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#automatically saved to the cache

print(olmo)
