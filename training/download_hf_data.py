from datasets import load_dataset
import os

# Option A: Small scale (Fast, for testing)
print("Downloading small-scale pretraining data (Streaming mode)...")
ds_iterable = load_dataset("vukrosic/blueberry-1B-pretrain", split="train", streaming=True)
# Materialize a subset for local experimentation
from datasets import Dataset
ds = Dataset.from_list(list(ds_iterable.take(25000)))

os.makedirs("processed_data/pretrain_dataset", exist_ok=True)
ds.save_to_disk("processed_data/pretrain_dataset")

# Option B: Large scale
# print("Downloading full pretraining data...")
# ds = load_dataset("vukrosic/blueberry-1B-pretrain")
# os.makedirs("processed_data/pretrain_full", exist_ok=True)
# ds.save_to_disk("processed_data/pretrain_full")

print("✅ Data Ready!")
