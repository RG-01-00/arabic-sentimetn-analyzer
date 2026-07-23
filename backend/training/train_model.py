import pandas as pd
import numpy as np
import torch
import subprocess  # ADD THIS
import time        # ADD THIS
import os

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)




# ============================================================
# 1. CONFIGURATION
# ============================================================

EXCEL_PATH = "./data/LSAHR_Balanced.xlsx"



# Existing model used ONLY as the starting point
STARTING_MODEL_DIR = "./models/best_bert_model_interrupted"

# New checkpoints for this training run
CHECKPOINT_DIR = "./models/continued_5k_checkpoints"

# New final model destination
OUTPUT_DIR = "./models/continued_5k_earlystop_model"

# Start at row 20,200
SUBSET_OFFSET = 20200

# Train on 5,000 rows
TRAIN_SUBSET_SIZE = 8000


USE_SUBSET = True

# ============================================================
# 2. TEMPERATURE MONITORING
# ============================================================

def check_gpu_temp():
    """Check GPU temperature and throttle if too hot"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        temp = int(result.stdout.strip())
        return temp
    except:
        return 0

def wait_if_hot(max_temp=82, pause_seconds=60):
    """Pause training if GPU gets too hot"""
    temp = check_gpu_temp()
    if temp > max_temp:
        print(f"hot GPU temperature {temp}°C - pausing for {pause_seconds} seconds to cool...")
        time.sleep(pause_seconds)
        return True
    return False


# ============================================================
# 2. LOAD DATA
# ============================================================


df = pd.read_excel(EXCEL_PATH)

print("Original dataset size:", len(df))
print(df.head())


# Keep only useful columns
df = df[["review", "user_rating"]]

# Remove empty reviews
df = df.dropna(subset=["review", "user_rating"])

# Convert rating to sentiment
def rating_to_sentiment(rating):
    if rating <= 4:
        return "negative"
    elif rating <= 6:
        return "neutral"
    else:
        return "positive"

df["label"] = df["user_rating"].apply(rating_to_sentiment)  # CREATE label column
# Convert string labels to integers
label_to_id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["label"].map(label_to_id)

# Rename review to text
df = df.rename(columns={"review": "text"})

# Now you have exactly what the BERT training code needs
df = df[["text", "label"]]

# NEW: Option to use subset of data
if USE_SUBSET:
    # Take a slice of the data (chunk of 40K)
    df = df.iloc[SUBSET_OFFSET:SUBSET_OFFSET + TRAIN_SUBSET_SIZE]
    print(f"\nUsing subset: samples {SUBSET_OFFSET} to {SUBSET_OFFSET + TRAIN_SUBSET_SIZE}")
    print("Subset size:", len(df))

torch.cuda.empty_cache()

# ============================================================
# 4. SHOW DATA DISTRIBUTION
# ============================================================

print("\nDataset distribution:")
print(df["label"].value_counts())

print("\nFinal dataset size:", len(df))


# ============================================================
# 5. SPLIT DATA
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=42,
    stratify=df["label"]
)

train_df, validation_df = train_test_split(
    train_df,
    test_size=0.1111,
    random_state=42,
    stratify=train_df["label"]
)

print("\nTraining samples:", len(train_df))
print("Validation samples:", len(validation_df))
print("Testing samples:", len(test_df))


# ============================================================
# 6. CONVERT TO HUGGINGFACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["text", "label"]],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[["text", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["text", "label"]],
    preserve_index=False
)




# ============================================================
# 7. LOAD EXISTING MODEL
# ============================================================

print("\nLoading tokenizer from existing model...")

tokenizer = AutoTokenizer.from_pretrained(
    STARTING_MODEL_DIR
)


print("\nLoading existing trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    STARTING_MODEL_DIR
)


print("\nExisting model loaded successfully.")

print(
    "The original model will NOT be modified."
)
# ============================================================
# 8. TOKENIZE TEXT
# ============================================================

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# Remove unnecessary text column
train_dataset = train_dataset.remove_columns(["text"])
validation_dataset = validation_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])


# ============================================================
# 9. METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)

    f1 = f1_score(
        labels,
        predictions,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "f1": f1
    }


# ============================================================
# 10. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    num_train_epochs=10,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,

    learning_rate=2e-5,

    weight_decay=0.01,

    eval_strategy="steps",
    save_strategy="steps",

    eval_steps=200,
    save_steps=200,

    load_best_model_at_end=True,

    metric_for_best_model="f1",
    greater_is_better=True,

    logging_steps=100,

    save_total_limit=2,

    report_to="none",

    gradient_accumulation_steps=1,

    dataloader_pin_memory=False,

    dataloader_num_workers=0,
)


# ============================================================
# 11. TRAINER
# ============================================================
# Custom Trainer with temperature monitoring
class TemperatureAwareTrainer(Trainer):
    def training_step(self, model, inputs ,  num_items_in_batch=None):
        # Check temperature before each step
        wait_if_hot(max_temp=82, pause_seconds=60)
        return super().training_step(model, inputs ,num_items_in_batch)

trainer = TemperatureAwareTrainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=2
        )
    ]
)


# ============================================================
# 12. TRAIN
# ============================================================

print("\nStarting training...")
print(f"Training on {len(train_dataset)} samples")

# Find the latest checkpoint automatically
last_checkpoint = None

if os.path.isdir(CHECKPOINT_DIR):
    last_checkpoint = get_last_checkpoint(CHECKPOINT_DIR)

if last_checkpoint:
    print(f"\n🔄 Resuming training from checkpoint:")
    print(last_checkpoint)

else:
    print("\n🚀 Starting new training session")

try:

    if last_checkpoint:
        trainer.train(
            resume_from_checkpoint=last_checkpoint
        )
    else:
        trainer.train()

except KeyboardInterrupt:

    print("\n⚠️ Training interrupted!")

    print("Saving current model safely...")

    trainer.save_model(
        OUTPUT_DIR + "_interrupted"
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR + "_interrupted"
    )

    print(
        f"Interrupted model saved to: "
        f"{OUTPUT_DIR}_interrupted"
    )

# ============================================================
# 13. FINAL TEST
# ============================================================

print("\nEvaluating model on unseen test data...")

test_results = trainer.evaluate(test_dataset)

print("\nTest Results:")
print(test_results)


# ============================================================
# 14. SAVE FINAL MODEL
# ============================================================

print("\nSaving final model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel successfully saved to:")
print(OUTPUT_DIR)