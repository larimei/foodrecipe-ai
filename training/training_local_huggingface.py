try:
    import transformers
    import datasets
    import evaluate
    from huggingface_hub import login
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "transformers", "datasets", "evaluate"])

import torch
print("CUDA available: ", torch.cuda.is_available())
print("CUDA device count: ", torch.cuda.device_count())

from huggingface_hub import login
login()

# Load dataset
from datasets import load_dataset
import numpy as np
import pandas as pd

food = load_dataset("food101", split="train")
food = food.train_test_split(test_size=0.2)

from transformers import AutoImageProcessor, DefaultDataCollator, ResNetForImageClassification, TrainingArguments, Trainer

checkpoint = "microsoft/resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

labels = food["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples


food = food.with_transform(transforms)

data_collator = DefaultDataCollator()

import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = ResNetForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="larimei/food-classification-ai-resnet-5e",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=food["train"],
    eval_dataset=food["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()