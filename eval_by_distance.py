import logging
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.special import softmax
from custom_models.drope_ntv2 import create_drope_ntv2_model
from safetensors.torch import load_file
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
FINETUNED_MODEL = "./ntv2-finetuned/checkpoint-3000/model.safetensors"
MAX_LEN = 2048
TASK_NAME = "variant_effect_causal_eqtl"

# Load model
model = create_drope_ntv2_model(MODEL_NAME, attention_type="nope", num_labels=2)
state_dict = load_file(FINETUNED_MODEL, device="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tokeniser
tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load dataset
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=TASK_NAME,
    sequence_length=MAX_LEN,
    trust_remote_code=True,
    split='test'
)

# Preprocess the dataset
def preprocess_function(examples):
    return tokeniser(
        examples["alt_forward_sequence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )

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

def compute_metrics(pred):
    labels = pred.label_ids
    # Handle output as tuple (logits,) or ModelOutput
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    probabilities = softmax(logits, axis=-1)[:, 1] # probability for class 1
    auroc = roc_auc_score(labels, probabilities)  # using sklearn
    cm = confusion_matrix(labels, preds)
    return {"accuracy": acc, "auroc": auroc, "f1": f1, "cm": cm}

# Training arguments
training_args = TrainingArguments(
    per_device_eval_batch_size=4,
    fp16=False,
    do_train=False,
    do_eval=True,
    report_to="none"
)

# Split dataset by distance to nearest TSS
df = dataset.to_pandas()

df_1 = df[df["distance_to_nearest_tss"].isin(list(range(0, 30000)))]
df_2 = df[df["distance_to_nearest_tss"].isin(list(range(30000, 100000)))]
df_3 = df[df["distance_to_nearest_tss"]>=100000]

for df in [df_1, df_2, df_3]:

    seqs = df['alt_forward_sequence'].to_list()
    labels = df['label'].to_list()
    encodings = tokeniser(seqs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')

    eval_dataset = VarDataset(encodings, labels)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    eval_results = trainer.evaluate()
    print(eval_results)

