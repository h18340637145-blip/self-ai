from datasets import load_dataset
import time

dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

iterator = iter(dataset)

first = next(iterator)
print(first)

count = 0

start_time = time.perf_counter()
for item in iterator:
    count += 1
    if time.perf_counter() - start_time >= 10:
        break
end_time = time.perf_counter()
print(f"Processed {count} items in {end_time - start_time} seconds.")