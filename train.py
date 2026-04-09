import torch
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Step 1: Initialization & Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./qwen-chat-finetune"
DATA_JSON = "./qustions_and_quaries.json"

# Hardware optimization: Distribute memory safely across dual GPUs
max_memory = {0: "13GiB", 1: "13GiB"}

# --- Step 2: Tokenizer & Model Instantiation ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory,
    trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# --- Step 3: LoRA Adapter Configuration ---
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,  
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# --- Step 4: Dataset Formatting & Tokenization ---
def format_data(example):
    system_part = "<|im_start|>system\nYou are a Text-to-SQL expert. Answer only with SQL queries.<|im_end|>\n"
    instruction_part = f"<|im_start|>user\n{example['input']}<|im_end|>\n"
    response_part = f"<|im_start|>assistant\n{example['query']}<|im_end|>"
    
    full_prompt = system_part + instruction_part + response_part
    tokenized_full = tokenizer(full_prompt, truncation=True, max_length=384, padding=False)
    
    # Mask instructions (-100) so the model only optimizes for the response generation
    instruction_len = len(tokenizer(system_part + instruction_part)["input_ids"])
    labels = [-100] * instruction_len + list(tokenized_full["input_ids"])[instruction_len:]
    tokenized_full["labels"] = labels
    
    return tokenized_full

dataset = load_dataset("json", data_files=DATA_JSON, split="train")
dataset = dataset.map(format_data, remove_columns=dataset.column_names)

# --- Step 5: Training Setup & Hyperparameters ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.05,
    label_smoothing_factor=0.05,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    report_to="none",
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

# --- Step 6: Execution & Model Persistence ---
torch.cuda.empty_cache()
trainer.train()

# Export final weights
final_dir = f"{OUTPUT_DIR}/final_adapter"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
print(f"✅ Training Complete. Adapter weights exported to: {final_dir}")