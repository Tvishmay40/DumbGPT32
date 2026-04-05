# config.py

# Neural Network Hyperparameters
BLOCK_SIZE = 64      # Maximum context length
EMBED_SIZE = 64      # Embedding dimension
NUM_HEADS = 4        # Number of attention heads
NUM_LAYERS = 2       # Number of transformer blocks
DROPOUT = 0.1

# Training Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_ITERS = 3000
EVAL_INTERVAL = 300
DEVICE = 'cpu'       # Change to 'cuda' if you have a GPU