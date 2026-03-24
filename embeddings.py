
import re
import string
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pprint import pprint
import json
from collections import Counter
import seaborn

from data import get_paragraphs
from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


text = get_paragraphs()

model, tokenizer = get_model()
inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
# This sets up a context in which PyTorch will not track gradients,
# which is useful for inference since we don't need to compute or store gradients.
with torch.no_grad():
    outputs = model(**inputs, use_cache=True, return_dict =True) 
past_key_values = outputs.past_key_values
# This is a tuple of (key, value) per layer
# Each key tensor has shape: (batch, num_kv_heads, seq_len, head_dim)

#To get all key embeddings for a specific token position across all layers:

token_pos = 5  # position of your token in the sequence
# Key embeddings for that token at every layer

#num_layers = 32
#batch = 1 (only one input string being used here)
#n/seq_len: 52
#d/head_dim: 128
#shape of .keys: (batch = 1, num_kv_heads = 32, seq_len = 52, head_dim = 128). 

all_keys = [past_key_values.layers[layer].keys[:, :, token_pos, :]
          for layer in range(len(past_key_values))]
# all_keys[i] has shape (1, num_kv_heads, head_dim)

token_ids = inputs["input_ids"][0]
target_id = tokenizer.encode("the", add_special_tokens=False)[0]
positions = (token_ids == target_id).nonzero(as_tuple=True)[0]


k1 = all_keys[0][0][0]
k2 = all_keys[0][0][1]



## making plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def get_polar_coordinate(a, b):
    '''
    a,b: vectors
    b: basic vector
    returns 
        - radius for a
        - angle btw a and b
    '''
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    cos_theta = torch.dot(a, b) / (na * nb)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0) # numerical stability
    theta = torch.arccos(cos_theta)  # radians
    r = na / nb
    return r, theta

#TODO find better way to set labels for polar coordinate plot (see 'k1k2_polar_test_plot.png')
#TODO show both basis and other vector?
def plot_polar_point(theta, r, save_path="plot.png"):
    theta, r = float(theta), float(r)
    if r < 0:
        raise ValueError("r must be non-negative")

    theta_deg = np.degrees(theta)
    x, y = r * np.cos(theta), r * np.sin(theta)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")

    # Draw vector from origin to (x, y)
    ax.quiver(
        0, 0, x, y,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="navy",
        width=0.008
    )

    # Draw angle arc
    arc_diameter = max(r * 0.4, 0.2)  # avoid degenerate arc when r is 0
    arc = mpatches.Arc( #creates arc
        (0, 0),
        arc_diameter,
        arc_diameter,
        angle=0,
        theta1=0,
        theta2=theta_deg,
        color="black",
        lw=1.5
    )
    ax.add_patch(arc)

    # Labels
    ax.text(x / 2 - 0.30, y / 2 + max(r * 0.08, 0.08), f"r = {r:.2f}", fontsize=12, color="navy")
    ax.text(max(r * 0.25, 0.15), max(r * 0.08, 0.08), f"θ = {theta_deg:.1f}°", fontsize=13)

    # Robust axis limits for all quadrants
    pad = max(r * 0.15, 0.2)
    x_min, x_max = min(0, x) - pad, max(0, x) + pad
    y_min, y_max = min(0, y) - pad, max(0, y) + pad
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_position("zero")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    
plot_polar_point(theta =np.pi/4, r =2, save_path = 'plot.png')