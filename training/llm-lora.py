

import torch
import pandas as pd
import numpy as np
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import Dataset
import evaluate
from peft import get_peft_model, LoraConfig, TaskType
from trl.trainer.sft_trainer import SFTTrainer

# --- 1. Cấu hình ---
parser = argparse.ArgumentParser(description='Configure LoRA training for LLMs.')
parser.add_argument('--model_name_or_path', type=str, default='Viet-Mistral/Vistral-7B-Chat', help='Model to fine-tune.')
parser.add_argument('--rationale_col', type=str, default='human_justification', help='Column name for rationale/justification.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Keep it small (1, 2, 4) for a 6GB GPU.')
parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs.')
args = parser.parse_args()

# --- 2. Tải Model và Tokenizer ---
print("--- Loading Model and Tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    load_in_8bit=True,
    use_safetensors=True
)

# --- 3. Cấu hình LoRA ---
print("\n--- Configuring LoRA ---")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# --- 4. Chuẩn bị Dữ liệu ---
print("\n--- Preparing Data ---")
# === SỬA LỖI ĐƯỜNG DẪN ===
# Sửa để đọc file train_rationale.xlsx từ thư mục data
train_df = pd.read_excel('data/train_rationale.xlsx') 
train_df[args.rationale_col] = train_df['human_justification'] # Giả định tên cột
train_df['label'] = train_df['label'].astype(str)
train_dataset = Dataset.from_pandas(train_df)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['text'])):
        text = f"""### Instruction:
Bạn là trợ lý nhận diện cảm xúc. Văn bản được cung cấp là lời nói trong lĩnh vực y tế. Hãy trả lời cảm xúc theo 3 nhãn: 0 (tiêu cực), 1 (trung tính), 2 (tích cực) và đưa ra lời giải thích ngắn gọn.

### Input:
{example['text'][i]}

### Response:
LABEL: {example['label'][i]} RATIONALE: {example[args.rationale_col][i]}"""
        output_texts.append(text)
    return output_texts

# --- 5. Huấn luyện ---
print("\n--- Setting up Training ---")
training_args = TrainingArguments(
    # === SỬA LỖI ĐƯỜNG DẪN ===
    output_dir=f"results/{args.model_name_or_path.split('/')[-1]}-lora",
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    bf16=True,
    num_train_epochs=args.num_train_epochs,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

print(f"\n--- Starting LoRA Training for {args.model_name_or_path} ---")
trainer.train()
trainer.save_model()