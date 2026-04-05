# prepare_dataset.py
import os
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
# Suggested: "lparkourer10/twitch_chat" for a larger, higher-quality dataset
DATASET_NAME = "roneneldan/TinyStories" 
SUBSET = ""
OUTPUT_FILE = "dataset.txt"
TARGET_SIZE_MB = 50  # Stop after writing this many MB of text

print(f"Loading {DATASET_NAME} [{SUBSET}]...")

# Load with streaming to save local disk space
ds = load_dataset(DATASET_NAME,split="train", streaming=True) #add another argument named ,SUBSET, if needed

print(f"Extracting text to {OUTPUT_FILE} (Target: {TARGET_SIZE_MB}MB)...")

text_key = None
bytes_written = 0
target_bytes = TARGET_SIZE_MB * 1024 * 1024

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(ds, desc="Writing Data"):
        # Automatically find which key contains the text on the first iteration
        if text_key is None:
            for key, value in item.items():
                if isinstance(value, str) and len(value) > 2:
                    text_key = key
                    print(f"Found text in column: '{text_key}'")
                    break
        
        if text_key and item[text_key]:
            line = str(item[text_key]).strip() + "\n"
            f.write(line)
            bytes_written += len(line.encode('utf-8'))
        
        # Stop once we reach the target file size
        if bytes_written >= target_bytes:
            break

print(f"\nSuccess! '{OUTPUT_FILE}' is now {bytes_written / (1024*1024):.2f} MB.")
print("Now you can proceed to 'python train.py'.")