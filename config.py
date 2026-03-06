
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MODEL = "allenai/Olmo-3-1025-7B"
DEFAULT_DATASET = "monology/pile-uncopyrighted"