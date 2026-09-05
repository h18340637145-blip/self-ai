from datasets import load_dataset

from pathlib import Path

dataset = load_dataset("nyu-mll/glue", "mrpc", split="train")

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

dataset.to_csv(output_dir / "mrpc_dataset.csv")
dataset.to_parquet(output_dir / "mrpc_dataset.parquet")


csv_size = (output_dir / "mrpc_dataset.csv").stat().st_size
parquet_size = (output_dir / "mrpc_dataset.parquet").stat().st_size

print(f"CSV size: {csv_size} bytes")
print(f"Parquet size: {parquet_size} bytes")