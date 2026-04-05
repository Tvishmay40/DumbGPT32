# prepare_dataset.py
from datasets import load_dataset
from tqdm import tqdm

# --- Configuration ---
DATASET_NAME = "lishysf/Twitch_Chat"
OUTPUT_FILE = "dataset.txt"
MAX_LINES = 1000000 # Limit to 1 million lines to prevent RAM crashes during training

print(f"Downloading dataset: {DATASET_NAME}...")
# Many datasets have a 'train' split by default. 
# We use streaming=True so it doesn't try to download 50GB to your hard drive all at once.
try:
    ds = load_dataset(DATASET_NAME, split="train", streaming=True)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying without specifying split...")
    ds = load_dataset(DATASET_NAME, streaming=True)
    # Get the first available split
    ds = ds[list(ds.keys())[0]]

print(f"Extracting text to {OUTPUT_FILE}...")

count = 0
# Open the file in write mode with UTF-8 encoding
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in tqdm(ds, desc="Writing lines"):
        # Datasets have different column names. 
        # Twitch chat usually uses 'message', 'body', or 'text'.
        # This checks for the most common ones.
        text_content = ""
        if "message" in item:
            text_content = str(item["message"])
        elif "text" in item:
            text_content = str(item["text"])
        elif "content" in item:
            text_content = str(item["content"])
        else:
            # Fallback: just grab the first column's data
            text_content = str(list(item.values())[0])

        # Clean up whitespace and ignore empty messages
        text_content = text_content.strip()
        if text_content: 
            f.write(text_content + "\n")
            count += 1
        
        # Stop if we hit our memory-safe limit
        if count >= MAX_LINES:
            break

print(f"\nSuccess! Wrote {count} lines to {OUTPUT_FILE}.")
print("You can now run train.py!")