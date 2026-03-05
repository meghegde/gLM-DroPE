import logging
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force Flash Attention / SDPA (critical for memory)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# Constants
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED_MODEL = "ntv2-500m-finetuned-2048bp"

MAX_LEN = 2048
TASK_NAME = "variant_effect_causal_eqtl"
BATCH_SIZE = 1

# Sliding window params (safe on H100)
WINDOW_SIZE = 8192
WINDOW_STRIDE = 4096

# Sliding Window Wrapper
class SlidingWindowWrapper(nn.Module):
    def __init__(self, base_model, window_size=8192, stride=4096):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.stride = stride

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, seq_len = input_ids.shape
        logits_list = []

        for start in range(0, seq_len, self.stride):
            end = min(start + self.window_size, seq_len)

            chunk_input_ids = input_ids[:, start:end]
            chunk_attention_mask = (
                attention_mask[:, start:end] if attention_mask is not None else None
            )

            outputs = self.base_model(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
                labels=None,  # ❗ do NOT compute loss per window
            )

            logits_list.append(outputs.logits)

            if end == seq_len:
                break

        # Aggregate logits
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

# Load base model + weights
logger.info("Loading base NT-v2 model")

base_model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL, ignore_mismatched_sizes=True)
base_model.eval()

# Wrap with sliding window inference
model = SlidingWindowWrapper(
    base_model,
    window_size=WINDOW_SIZE,
    stride=WINDOW_STRIDE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device, dtype=torch.bfloat16)
model.eval()
torch.set_grad_enabled(False)

# Tokenizer
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split="test",
)

def preprocess_function(examples):
    return tokeniser(
        examples["alt_forward_sequence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

encoded_dataset = dataset.map(preprocess_function, batched=True)
encoded_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"],
)

# Data collator
data_collator = DataCollatorWithPadding(tokeniser)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    logits = (
        pred.predictions[0]
        if isinstance(pred.predictions, tuple)
        else pred.predictions
    )

    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    probs = softmax(logits, axis=-1)[:, 1]
    auroc = roc_auc_score(labels, probs)

    cm = confusion_matrix(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "confusion_matrix": cm,
    }

# Trainer
training_args = TrainingArguments(
    per_device_eval_batch_size=BATCH_SIZE,
    do_train=False,
    do_eval=True,
    report_to="none",
    output_dir="./results",
    prediction_loss_only=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset,
    compute_metrics=compute_metrics,
)

# Evaluate
logger.info("Starting evaluation with sliding window inference")
eval_results = trainer.evaluate()
print(eval_results)
