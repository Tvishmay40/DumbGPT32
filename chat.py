# chat.py
import torch
import pickle
from model import MicroLLM
import config

# Load Vocab
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
stoi, itos, vocab_size = vocab['stoi'], vocab['itos'], vocab['vocab_size']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load Model
model = MicroLLM(vocab_size).to(config.DEVICE)
model.load_state_dict(torch.load('model.bin', map_location=config.DEVICE))
model.eval()

print("Model loaded. Type your prompt (or 'quit' to exit):")
while True:
    prompt = input(">> ")
    if prompt.lower() == 'quit':
        break
    
    context = torch.tensor((encode(prompt)), dtype=torch.long, device=config.DEVICE).unsqueeze(0)
    generated = model.generate(context, max_new_tokens=100)
    print(decode(generated[0].tolist()))