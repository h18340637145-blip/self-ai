from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb", split="train")

dataset.to_csv("doc/imdb_train.csv")
dataset.to_json("doc/imdb_train.json")
dataset.to_parquet("doc/imdb_train.parquet")