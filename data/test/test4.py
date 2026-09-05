from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", split="train")

split = dataset.train_test_split(test_size=0.3, seed=42)
val_val = split["test"].train_test_split(test_size=0.5, seed=42)

train_ds = split["train"]
val_ds = val_val["test"]
test_ds = val_val["train"]

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")