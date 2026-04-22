#libraries
from collections import Counter
from datasets import load_dataset
from huggingface_hub import HfFileSystem
import io
import json
import zstandard as zstd
from transformers import AutoModelForCausalLM, AutoTokenizer

#local files
from config import DEFAULT_DATASET, MAX_TOKENS
from model import get_tokenizer


PILE_COMPONENTS = [
    "Pile-CC",
    "PubMed Central",
    "ArXiv",
    "GitHub",
    "FreeLaw",
    "Stack Exchange",
    "USPTO Backgrounds",
    "DM Mathematics",
    "Wikipedia (en)",
    "HackerNews",
    "NIH ExPorter",
    "PubMed Abstracts",
    "Enron Emails",
    "EuroParl",
    "PhilPapers",
    "Ubuntu IRC",
    "Gutenberg (PG-19)",
]

def get_data(component="all", dataset_name=DEFAULT_DATASET):
    """
    Return a streaming IterableDataset for the Pile (or a single component).

    Args:
        component: one of PILE_COMPONENTS (e.g. "Wikipedia (en)") to restrict the
                   stream to that subset, or "all" (default) for the full dataset.
        dataset_name: HuggingFace dataset identifier.

    Returns:
        datasets.IterableDataset — a lazy stream; rows are fetched and decoded
        on demand, never all at once.

    Why streaming is beneficial:
        Streaming allows you to process massive datasets that do not fit into memory,
        since samples are loaded one at a time (or in small batches), rather than all at once.
        This enables working with datasets like the Pile efficiently, avoiding long initial
        downloads or memory errors, and supports true lazy iteration for scalable preprocessing
        and model evaluation.

    Performance note — why .filter() is faster than a Python for-loop:
        The HuggingFace datasets library processes rows in Arrow columnar batches.
        When .filter() evaluates the predicate it operates on a whole batch of
        "meta" values at once using vectorized C++ (via PyArrow), rather than
        calling a Python if-statement once per row. The "text" field is still
        read from disk, but the filtering logic itself runs in compiled code,
        giving a meaningful throughput improvement over a plain Python loop.
    """
    stream = load_dataset(dataset_name, split="train", streaming=True)
    if component != "all":
        stream = stream.filter(lambda x: x["meta"]["pile_set_name"] == component)
    return stream

def get_data_samples(
    component="all",
    max_tokens=20000,
    type="string",
    separator="\n\n",
    dataset_name=DEFAULT_DATASET,
    return_metadata=False,
):
    """
    Collect samples from the dataset until a token budget is exhausted.

    Streams samples one at a time, tokenizing each and accumulating them until
    adding the next sample would exceed `max_tokens`. The sample that would
    exceed the limit is skipped and iteration stops.

    Args:
        component (str): Dataset component to filter on, or "all" for the full dataset.
        max_tokens (int): Maximum total number of tokens to collect across all samples.
        type (str): Output format — "array" returns a list of strings, "string" (default)
                    joins all samples with `separator`.
        separator (str): Separator used when `type` is "string".
        dataset_name (str): HuggingFace dataset identifier.
        return_metadata (bool): If True, return a tuple of (result, metadata) where
                                metadata is a dict with num_samples, num_tokens,
                                num_words, and avg_tokens_per_sample.

    Returns:
        list[str] | str: The collected samples as a list or joined string, depending on `type`.
        tuple[list[str] | str, dict]: If `return_metadata` is True, a tuple of the above
                                      result and a metadata dict with keys:
                                        - num_samples (int)
                                        - num_tokens (int)
                                        - num_words (int)
                                        - avg_tokens_per_sample (float)
    """
    tokenizer = get_tokenizer()
    stream = get_data(component=component, dataset_name=dataset_name)
    samples = []
    total_tokens = 0
    total_words = 0
    for i, sample in enumerate(stream):
        text = sample["text"]
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if total_tokens + n_tokens > max_tokens:
            break

        if ((i +1) % 100 ==0):
            print(f"sample {i}") #print multiples of 100
            
        samples.append(text)
        total_tokens += n_tokens
        total_words += len(text.split())
    print(f"Collected {len(samples)} samples using {total_tokens}/{max_tokens} tokens.")

    metadata = {
        "num_samples": len(samples),
        "num_tokens": total_tokens,
        "num_words": total_words,
        "avg_tokens_per_sample": total_tokens / len(samples) if samples else 0,
    }

    if type == "string":
        result = separator.join(samples)
    else:
        result = samples

    if return_metadata:
        return result, metadata

    return result



#### OLD FUNCTIONS

# n: how many paragraphs of text in string
# start: starting paragraph
#NOTE this is not perfect as not all \n\n indicate paragraphs, eg. \n\nConcept\n\n
def get_paragraphs(n=1, start_idx=0, component="all"):
    end_idx = start_idx + n
    data = get_data(component=component)
    sample = next(iter(data))
    paragraphs_split = sample['text'].split("\n\n")[start_idx:end_idx]
    paragraphs = "\n\n".join(paragraphs_split)
    return paragraphs

# data + model
def get_data_model():
    data = get_data()
    model, tokenizer = get_model()
    return (data, model, tokenizer)

## OLD
def get_data_sample(sample_idx=0, component="all"):
    data = get_data(component=component)
    iterator = iter(data)
    for _ in range(sample_idx):
        next(iterator)
    sample = next(iterator)
    return sample['text']


def get_data_samples_range(
    start=0,
    end=1,
    component="all",
    type="string",
    separator="\n\n",
    dataset_name=DEFAULT_DATASET,
):
    """
    Retrieve samples from a specified component (or all components) of the dataset.

    Args:
        start (int): The index of the first sample to retrieve (inclusive).
        end (int): The index one past the last sample to retrieve (exclusive). If -1, retrieve to the end of the stream.
        component (str): Dataset component to filter on, or "all" for the full dataset.
        type (str): Output type, either "array" for a list of strings or "string" (default) to join samples by `separator`.
        separator (str): Separator string used if `type` is "string".
        dataset_name (str): HuggingFace dataset identifier.

    Returns:
        list[str] or str: The selected samples, either as a list ("array") or a single string ("string", default) with the given separator.
    """
    stream = get_data(component=component, dataset_name=dataset_name)
    samples = []
    for i, sample in enumerate(stream):
        if i < start:
            continue
        if end != -1 and i >= end:
            break
        
        if ((i +1) % 100 ==0):
            print(f"sample {i}") #print multiples of 100

        samples.append(sample["text"])

    if end != -1 and len(samples) < (end - start):
        print(f"Warning: stream exhausted early — expected {end - start} samples but only found {len(samples)}.")

    if type == "array":
        return samples
    return separator.join(samples)