import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import transformers
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score

# Random seed
SEED = 42
transformers.set_seed(SEED)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
# https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species
# Pretrained on 2048bp sequences
MAX_LEN = 2048 # batch_size=4
TASK_NAME = "variant_effect_causal_eqtl"
OUTPUT_DIR = f"./ntv2-{MAX_LEN}-bp"
BATCH_SIZE = 4

# Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load tokeniser
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split='train'
)
df = pd.DataFrame.from_dict(dataset)
seqs = df['alt_forward_sequence'].to_list()
labels = df['label'].to_list()

# Split into training and validation sets
train_seqs, val_seqs, train_labels, val_labels = train_test_split(seqs, labels, test_size=.2, shuffle=True, random_state=SEED)

# Tokenise data
train_encodings = tokeniser(train_seqs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
val_encodings = tokeniser(val_seqs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')

# Adapted from https://huggingface.co/transformers/v3.2.0/custom_datasets.html
class VarDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = VarDataset(train_encodings, train_labels)
val_dataset = VarDataset(val_encodings, val_labels)

# Define compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    num_train_epochs=100,  # high, early stopping will stop earlier
    # save_strategy="steps",
    # save_steps=1000,
    save_total_limit=1,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Trainer - stop if val_loss does not improve after 3 epochs
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Save logs
log_file = OUTPUT_DIR+'/training_log.txt'
history = trainer.state.log_history
logSave = open(log_file, 'w')
logSave.write(str(history))
logSave.close()

