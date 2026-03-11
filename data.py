#libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

#local files
from config import DEFAULT_DATASET
from model import get_model
#data only
def get_data(dataset_name=DEFAULT_DATASET):
    data = load_dataset(dataset_name, streaming=True)
    return data

def get_data_sample():
    data = get_data()
    sample = next(iter(data['train']))
    return sample['text']

# n: how many paragraphs of text in string
# start: starting paragraph
#NOTE this is not perfect as not all \n\n indicate paragraphs, eg. \n\nConcept\n\n
def get_paragraphs(n = 1, start_idx = 0):
    end_idx = start_idx + n
    data = get_data()       
    sample = next(iter(data['train']))
    paragraphs_split = sample['text'].split("\n\n")[start_idx:end_idx] #gets the first  to nth paragraph
    paragraphs = "\n\n".join(paragraphs_split)
    return paragraphs

# data + model
def get_data_model():
    data = get_data()
    model, tokenizer = get_model()
    return (data, model, tokenizer)
