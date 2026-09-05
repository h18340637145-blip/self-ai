from datasets import load_dataset

dataset = load_dataset("nyu-mll/glue", "mrpc")
print(dataset)

for i in range(5):
    print(dataset["train"][i])