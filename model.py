import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DEFAULT_MODEL


def _ensure_model_backend_available(model_name):
    # Surface the real backend import problem instead of the generic lazy-loader error.
    if "Olmo-3" not in model_name and "olmo-3" not in model_name:
        return

    try:
        from transformers import Olmo3ForCausalLM  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The selected model requires transformers support for OLMo3, but the current "
            "Python process could not import `Olmo3ForCausalLM`.\n"
            "Likely causes:\n"
            "1. You are running a stale notebook/kernel after changing packages.\n"
            "2. The interpreter running this code is not the same `.venv` where you installed "
            "`transformers`.\n"
            "3. Your `transformers` install is incomplete or incompatible with this model.\n"
            f"model_name={model_name!r}"
        ) from exc


def get_model(model_name=DEFAULT_MODEL):
    _ensure_model_backend_available(model_name)
    # device_map="auto" streams weights directly to GPU via accelerate,
    # avoiding the CPU staging copy that would require 2x model size in RAM.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
        torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_attentions(input_data, model, tokenizer):
    '''
    from input string, run inference and get the attention scores of the model
    '''
    inputs = tokenizer(input_data, return_tensors="pt")
    # Move inputs to the same device as the model's first parameter.
    # device_map="auto" may place the model on a device other than DEVICE.
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # Cast to float16 immediately to halve CPU memory when callers move tensors
    # off the GPU. Relative attention scores don't need float32 precision.
    return tuple(a.half() for a in outputs.attentions)
