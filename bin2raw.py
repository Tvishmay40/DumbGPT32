# export.py
import torch
import struct
from model import MicroLLM
import pickle

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

model = MicroLLM(vocab['vocab_size'])
model.load_state_dict(torch.load('model.bin', map_location='cpu'))

# Write weights sequentially as 32-bit floats
with open('model.raw', 'wb') as f:
    for name, param in model.named_parameters():
        # Flatten the tensor to a 1D array of floats
        weights = param.detach().numpy().flatten()
        # Pack floats into bytes (little-endian)
        f.write(struct.pack(f'<{len(weights)}f', *weights))

print("Exported raw weights to model.raw")
print(f"Total weights file size: {f.tell() / 1024:.2f} KB")