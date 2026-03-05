import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
import transformers
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from custom_models.drope_ntv2 import create_drope_ntv2_model

# Set random seed
#SEED = 42
SEED = 41
transformers.set_seed(SEED)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
MAX_LEN = 2048
ATTENTION_TYPE = "qk_norm_nope"
LEARNING_RATE = 5e-5
TASK_NAME = "variant_effect_causal_eqtl"
OUTPUT_DIR = f"./ntv2-{ATTENTION_TYPE}-lr={LEARNING_RATE}-{MAX_LEN}-bp-seed={SEED}"
NUM_LABELS = 2
BATCH_SIZE = 4

# Load model
model = create_drope_ntv2_model(MODEL_NAME, attention_type=ATTENTION_TYPE, num_labels=NUM_LABELS)
logger.info("Loaded DROPE NT-v2 model for classification")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
from datasets import load_dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split="train"
)

df = pd.DataFrame(dataset)
seqs = df['alt_forward_sequence'].tolist()
labels = df['label'].tolist()

train_seqs, val_seqs, train_labels, val_labels = train_test_split(
    seqs, labels, test_size=0.2, shuffle=True, random_state=SEED
)

# Tokenisation
train_encodings = tokenizer(
    train_seqs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
)
val_encodings = tokenizer(
    val_seqs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt"
)

# Custom Dataset
class VarDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Important for CE loss
        return item

train_dataset = VarDataset(train_encodings, train_labels)
val_dataset = VarDataset(val_encodings, val_labels)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=100,
    save_strategy="steps",
    save_steps=1000,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save logs
log_file = OUTPUT_DIR + ".txt"
with open(log_file, "w") as f:
    f.write(str(trainer.state.log_history))

logger.info("Training finished and logs saved")
