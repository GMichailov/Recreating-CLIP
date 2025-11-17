from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
import torch

print("CUDA:", torch.cuda.is_available())

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

ds = load_dataset("laion/laion400m", split="train", streaming=True)
print(next(iter(ds)))
