
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


text = get_paragraphs(5)

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

surv_pos = 11  # position of your token in the sequence
ival_pos = 12
# Key embeddings for that token at every layer

#num_layers = 32
#batch = 1 (only one input string being used here)
#n/seq_len: 52
#d/head_dim: 128
#shape of .keys: (batch = 1, num_kv_heads = 32, seq_len = 52, head_dim = 128). 

all_keys_1 = [past_key_values.layers[layer].keys[:, :, surv_pos, :]
          for layer in range(len(past_key_values))]


all_keys_2 = [past_key_values.layers[layer].keys[:, :, ival_pos, :]
          for layer in range(len(past_key_values))]

# all_keys has shape (num_layers, 1, num_kv_heads, head_dim)
# 1 b/c 1 seq took
# len(all_keys) = 32


k1 = all_keys_1[0][0][0]
k2 = all_keys_2[0][0][0]



## making plots
import numpy as np
import matplotlib.pyplot as plt

def get_polar_coordinates(a, b):
    '''
    a,b: vectors
    b: basic vector
    returns 
        - radius for a
        - angle btw a and b (in radians)
    '''
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    cos_theta = torch.dot(a, b) / (na * nb)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0) # numerical stability
    theta = torch.arccos(cos_theta)  # radians
    r = na / nb
    return r, theta

r,theta = get_polar_coordinates(k1,k2)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('module://matplotlib-backend-kitty')

def plot_polar_point(r, theta, save_path="plot.png"):
    """
    Plot a single polar coordinate (r, theta) where theta is in radians.
    Labels for r and theta remain readable even at small angles.
    """
    r, theta = float(r), float(theta)
    theta_deg = np.degrees(theta)

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection="polar"), facecolor="white")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetalim(0, np.pi)

    r_max = max(r * 1.35, 1.0)
    ax.set_ylim(0, r_max)
    ax.set_thetagrids([0, 45, 90, 135, 180])
    ax.set_rlabel_position(90)

    # Vector line + endpoint dot
    ax.plot([0, theta], [0, r], color="navy", lw=2.5, zorder=3)
    ax.scatter([theta], [r], color="navy", s=25, zorder=4)

    # Angle arc
    r_arc = r * 0.25
    theta_arc = np.linspace(0, theta, 100)
    ax.plot(theta_arc, np.full_like(theta_arc, r_arc), color="black", lw=1.2)

    #radius and theta labels
    label_r = r_arc * (2.5 if theta_deg < 20 else 1.6)
    r_label_offset = r_max * (0.12 if theta_deg > 20 else 0.18)
    ax.text(
        theta, r * 0.5, f"({r:.2f}, {theta_deg:.2f}°)",
        ha="center", va="bottom", fontsize=11, color="navy", fontweight="bold",
        rotation=theta_deg, rotation_mode="anchor"
    )

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# plot_polar_point(r=2.5, theta=np.radians(60), save_path="exmaple.png")
plot_polar_point(r,theta,'survival_polar.png')