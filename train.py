# train.py
import torch
import pickle
from model import MicroLLM
import config

# Load Data
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Save vocab for inference
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos, 'vocab_size': vocab_size}, f)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data_split[i:i+config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+config.BLOCK_SIZE+1] for i in ix])
    return x.to(config.DEVICE), y.to(config.DEVICE)

model = MicroLLM(vocab_size).to(config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e3} K")

for iter in range(config.MAX_ITERS):
    if iter % config.EVAL_INTERVAL == 0:
        print(f"Step {iter} complete...")
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model
torch.save(model.state_dict(), 'model.bin')
print("Training complete. Saved to model.bin")