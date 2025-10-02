

import pandas as pd
import numpy as np
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import evaluate
import torch

# --- 1. Cấu hình ---
parser = argparse.ArgumentParser(description='Configure training parameters.')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training. Start with 16 for a 6GB GPU.')
parser.add_argument('--num_train_epochs', type=int, default=15, help='Number of epochs for training.')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
parser.add_argument('--model_checkpoint', type=str, default="VietAI/vit5-base", help='Seq2Seq model checkpoint to use.')
args = parser.parse_args()

# --- 2. Tải Dữ liệu ---
print("--- Loading Data ---")
# === SỬA LỖI ĐƯỜNG DẪN ===
train_df = pd.read_excel('data/train.xlsx')
train_df['label'] = train_df['label'].astype(str)
train_dataset = Dataset.from_pandas(train_df)

testset = pd.read_excel('data/test.xlsx')
testset['label'] = testset['label'].astype(str)
test_dataset = Dataset.from_pandas(testset[['text', 'label']])

print(f"Training data labels: {train_df['label'].unique()}")
print(f"Test data labels: {testset['label'].unique()}")

# --- 3. Tiền xử lý ---
print("\n--- Initializing Model and Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint,use_safetensors=True)

def preprocess_function(examples):
    inputs = [doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)
    labels = tokenizer(text_target=examples["label"], max_length=8, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("\n--- Tokenizing Data ---")
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
tokenized_train_dataset = tokenized_train_dataset.remove_columns(['text', 'label'])
tokenized_test_dataset = tokenized_test_dataset.remove_columns(['text', 'label'])

# --- 4. Hàm Đánh giá (Metrics) ---
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() if pred.strip().isdigit() else "-1" for pred in decoded_preds]
    decoded_labels = [label.strip() if label.strip().isdigit() else "-1" for label in decoded_labels]

    acc_result = accuracy.compute(predictions=decoded_preds, references=decoded_labels)
    if acc_result:
        return {"accuracy": acc_result['accuracy']}
    return {"accuracy": 0.0}

# --- 5. Huấn luyện ---
print("\n--- Setting up Training ---")
model_name = args.model_checkpoint.split("/")[-1]

training_args = Seq2SeqTrainingArguments(
    output_dir=f"results/{model_name}-seq2seq",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=2,
    bf16=True,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    predict_with_generate=True,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model='accuracy',
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
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