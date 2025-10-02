import pandas as pd
import numpy as np
import argparse
import transformers
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
import evaluate
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

training_args = TrainingArguments(
    output_dir=f"results/{model_name}-encoder",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=2,
    bf16=True,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model='macro_f1',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)]
)

print(f"\n--- Starting Training for {args.model_checkpoint} ---")
trainer.train()
trainer.save_model()
print("\n--- Evaluating Final Model ---")
trainer.evaluate()