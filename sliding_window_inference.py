import logging
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
from safetensors.torch import load_file
import torch.nn.functional as F

from custom_models.drope_ntv2 import create_drope_ntv2_model

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force Flash Attention / SDPA (critical for memory)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# Constants
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED_MODEL = "ntv2-qk-norm-nope-lr=5e-5-1000-bp/model.safetensors"

MAX_LEN = 2048*5
TASK_NAME = "variant_effect_causal_eqtl"
ATTENTION_TYPE = "qk_norm_nope"
BATCH_SIZE = 1

# Sliding window params
WINDOW_SIZE = 8192
WINDOW_STRIDE = 4096

# Sliding Window Wrapper
class SlidingWindowWrapper(nn.Module):
    """
    Applies sliding-window inference for long sequences and
    aggregates window-level logits by averaging.
    """

    def __init__(self, base_model, window_size=8192, stride=4096):
        super().__init__()
        self.base_model = base_model
        self.window_size = window_size
        self.stride = stride

    def forward(self, input_ids, attention_mask=None, labels=None):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        chunks_input_ids = []
        chunks_attention_mask = []

        for start in range(0, seq_len, self.stride):
            end = min(start + self.window_size, seq_len)

            chunk_ids = input_ids[:, start:end]

            # Pad last window to fixed window_size
            pad_len = self.window_size - chunk_ids.shape[1]
            if pad_len > 0:
                chunk_ids = F.pad(chunk_ids, (0, pad_len), value=0)

            chunks_input_ids.append(chunk_ids)

            if attention_mask is not None:
                chunk_mask = attention_mask[:, start:end]
                if pad_len > 0:
                    chunk_mask = F.pad(chunk_mask, (0, pad_len), value=0)
                chunks_attention_mask.append(chunk_mask)

            if end == seq_len:
                break

        chunks_input_ids = torch.cat(chunks_input_ids, dim=0)

        if attention_mask is not None:
            chunks_attention_mask = torch.cat(chunks_attention_mask, dim=0)

        outputs = self.base_model(
            input_ids=chunks_input_ids,
            attention_mask=chunks_attention_mask if attention_mask is not None else None,
            labels=None
        )

        logits = outputs.logits

        num_windows = logits.shape[0] // batch_size
        logits = logits.view(num_windows, batch_size, -1).mean(dim=0)

        return logits

# Load base model + weights
logger.info("Loading base NT-v2 model")

base_model = create_drope_ntv2_model(
    MODEL_NAME,
    attention_type=ATTENTION_TYPE,
    num_labels=2,
)

state_dict = load_file(FINETUNED_MODEL, device="cpu")
base_model.load_state_dict(state_dict, strict=False)
base_model.eval()

# Wrap with sliding window inference
model = SlidingWindowWrapper(
    base_model,
    window_size=WINDOW_SIZE,
    stride=WINDOW_STRIDE,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device, dtype=torch.bfloat16)

# Tokeniser
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
    fp16=False,          # using BF16 manually
    do_train=False,
    do_eval=True,
    report_to="none",
    output_dir="./results",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset,
    tokenizer=tokeniser,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate
logger.info("Starting evaluation with sliding window inference")
eval_results = trainer.evaluate()
print(eval_results)
