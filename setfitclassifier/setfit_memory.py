from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from datasets import load_dataset
import torch
import gc

def memory_stats():
    return f"\nMemory allocated: {torch.cuda.memory_allocated()/1024**2}\nMemory reserved: {torch.cuda.memory_reserved()/1024**2}"

# Initializing a new SetFit model
model = SetFitModel.from_pretrained("BAAI/bge-small-en-v1.5", labels=["negative", "positive"])

# Preparing the dataset
dataset = load_dataset("SetFit/sst2")
train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=8)
test_dataset = dataset["test"]

# Preparing the training arguments
args = TrainingArguments(
    batch_size=32,
    num_epochs=10,
)

# Preparing the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
)
trainer.train()

# Evaluating
metrics = trainer.evaluate(test_dataset)
print(metrics)
# => {'accuracy': 0.8511806699615596}

print('\nMemory stats before release:',memory_stats())

del trainer
del model
gc.collect()
torch.cuda.empty_cache()

print('\nMemory stats after release:',memory_stats())

