import pandas as pd
import numpy as np
import argparse
import torch
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import evaluate
from inspect import signature
from typing import Any, Dict
# Initialize argparse
parser = argparse.ArgumentParser(description='Configure training parameters.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training. Start with 32 for a 6GB GPU.')
parser.add_argument('--num_train_epochs', type=int, default=15, help='Number of epochs for training.')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('--model_checkpoint', type=str, default="VietAI/vit5-base", help='Model checkpoint to use.')
args = parser.parse_args()

# --- 1. Tải Dữ liệu ---
print("--- Loading Data ---")
id2label = {0: "negative", 1: "neutral", 2: "positive"}
label2id = {"negative": 0, "neutral": 1, 'positive': 2}

# === SỬA LỖI ĐƯỜNG DẪN ===
# Chạy script từ thư mục gốc của project (D:/Sentiment-Reasoning-Project/)
train_df = pd.read_excel('data/train.xlsx') 
train_dataset = Dataset.from_pandas(train_df)

testset = pd.read_excel('data/test.xlsx')
test_dataset = Dataset.from_pandas(testset[['text', 'label']])

print(f"Training data labels: {train_df['label'].unique()}")
print(f"Test data labels: {testset['label'].unique()}")

# --- 2. Tiền xử lý ---
print("\n--- Initializing Model and Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_checkpoint, num_labels=3, id2label=id2label, label2id=label2id, use_safetensors=True
)

# Explicitly move model to GPU if available
if torch.cuda.is_available():
    gpu_idx = torch.cuda.current_device()
    model.to(f"cuda:{gpu_idx}")
    try:
        gpu_name = torch.cuda.get_device_name(gpu_idx)
    except Exception:
        gpu_name = f"cuda:{gpu_idx}"
    print(f"[Device] Using GPU: {gpu_name}")
else:
    print("[Device] CUDA not available. Training will run on CPU.")

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=256)

print("\n--- Tokenizing Data ---")
tokenized_dataset_train = train_dataset.map(preprocess_function, batched=True)
tokenized_dataset_test = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 3. Hàm Đánh giá (Metrics) ---
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    acc_result = accuracy.compute(predictions=predictions, references=labels)
    macro_f1_result = f1.compute(predictions=predictions, references=labels, average='macro')
    f1_all_result = f1.compute(predictions=predictions, references=labels, average=None)

    metrics_result = {}
    if acc_result:
        metrics_result["accuracy"] = acc_result['accuracy']
    if macro_f1_result:
        metrics_result["macro_f1"] = macro_f1_result['f1']
    if f1_all_result and 'f1' in f1_all_result and len(f1_all_result['f1']) == 3:
        metrics_result["f1_neg"] = f1_all_result['f1'][0]
        metrics_result["f1_neu"] = f1_all_result['f1'][1]
        metrics_result["f1_pos"] = f1_all_result['f1'][2]
        
    return metrics_result

# --- 4. Huấn luyện ---
print("\n--- Setting up Training ---")
model_name = args.model_checkpoint.split("/")[-1]

print(f"Transformers version: {transformers.__version__}")

# Decide precision based on hardware support
has_cuda = torch.cuda.is_available()
try:
    supports_bf16 = has_cuda and bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
except Exception:
    # Fallback heuristic: Ampere (SM80) or newer generally supports bf16
    try:
        major_cc, _ = torch.cuda.get_device_capability(0) if has_cuda else (0, 0)
        supports_bf16 = has_cuda and major_cc >= 8
    except Exception:
        supports_bf16 = False

print(f"CUDA available: {has_cuda} | BF16 supported: {supports_bf16}")

# Build TrainingArguments kwargs in a version-safe way
ta_kwargs: Dict[str, Any] = dict(
    output_dir=f"results/{model_name}-encoder",
    # Newer API keys (will be filtered out if unsupported)
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    # Common keys
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=2,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model='macro_f1',
    # GPU-friendly dataloader setting (safe on CUDA, ignored if unsupported)
    dataloader_pin_memory=True,
    # Enable TF32 on Ampere+ for speed (ignored if unsupported)
    tf32=True,
    # Older API fallbacks (ignored by newer versions)
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
)

# Precision flags (GPU mixed precision)
if supports_bf16:
    ta_kwargs['bf16'] = True
elif has_cuda:
    ta_kwargs['fp16'] = True

sig = signature(TrainingArguments.__init__)
supported_params = set(sig.parameters.keys())
filtered_kwargs: Dict[str, Any] = {k: v for k, v in ta_kwargs.items() if k in supported_params}
# Fallback: some older versions used 'eval_strategy' instead of 'evaluation_strategy'
if 'evaluation_strategy' not in supported_params and 'eval_strategy' in supported_params and 'evaluation_strategy' in ta_kwargs:
    filtered_kwargs['eval_strategy'] = ta_kwargs['evaluation_strategy']

# CPU fallback: prefer 'use_cpu' if available, else 'no_cuda'
if not has_cuda:
    if 'use_cpu' in supported_params:
        filtered_kwargs['use_cpu'] = True
    elif 'no_cuda' in supported_params:
        filtered_kwargs['no_cuda'] = True
skipped = sorted(set(ta_kwargs.keys()) - set(filtered_kwargs.keys()))

# If bf16 is not supported but fp16 is, fall back to fp16
if 'bf16' not in supported_params and ta_kwargs.get('bf16') is True and 'fp16' in supported_params:
    filtered_kwargs['fp16'] = True

if skipped:
    print(f"TrainingArguments: skipping unsupported args: {skipped}")

training_args = TrainingArguments(**filtered_kwargs)  # pyright: ignore[reportGeneralTypeIssues]

callbacks_list = []
if hasattr(transformers, "EarlyStoppingCallback"):
    callbacks_list.append(transformers.EarlyStoppingCallback(early_stopping_patience=3))
else:
    print("EarlyStoppingCallback not available; continuing without early stopping.")

# Trainer kwargs are version-gated
trainer_sig = signature(Trainer.__init__)
trainer_supported = set(trainer_sig.parameters.keys())
trainer_kwargs: Dict[str, Any] = {}
trainer_kwargs['model'] = model
trainer_kwargs['args'] = training_args
if 'train_dataset' in trainer_supported:
    trainer_kwargs['train_dataset'] = tokenized_dataset_train
if 'eval_dataset' in trainer_supported:
    trainer_kwargs['eval_dataset'] = tokenized_dataset_test
if 'tokenizer' in trainer_supported:
    trainer_kwargs['tokenizer'] = tokenizer
if 'data_collator' in trainer_supported:
    trainer_kwargs['data_collator'] = data_collator
if 'compute_metrics' in trainer_supported:
    trainer_kwargs['compute_metrics'] = compute_metrics
if 'callbacks' in trainer_supported and callbacks_list:
    trainer_kwargs['callbacks'] = callbacks_list

unsupported_trainer_keys = {'tokenizer', 'data_collator', 'compute_metrics', 'callbacks'} - trainer_supported
# Suppress noise for 'tokenizer' which is optional in many versions
to_report = unsupported_trainer_keys - {'tokenizer'}
if to_report:
    print(f"Trainer: skipping unsupported args: {sorted(list(to_report))}")

trainer = Trainer(**trainer_kwargs)

print(f"\n--- Starting Training for {args.model_checkpoint} ---")
print(f"[Device] torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Device] Trainer device: {trainer.args.device} | n_gpu={getattr(trainer.args,'n_gpu','NA')}")
else:
    print(f"[Device] Trainer device: {trainer.args.device}")
trainer.train()
trainer.save_model()
print("\n--- Evaluating Final Model ---")
trainer.evaluate()