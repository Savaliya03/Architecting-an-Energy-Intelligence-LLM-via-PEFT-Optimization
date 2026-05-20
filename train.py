### traing

# --- Step 1: Imports & Libraries ---
import torch
import os
from datasets import load_dataset

# transformers: The core library to load the AI model and run the training loop
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForSeq2Seq, BitsAndBytesConfig
)

# peft: Parameter-Efficient Fine-Tuning. Allows us to train only a tiny fraction of the model's weights.
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Step 2: Initialization & Configuration ---
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"  # The base foundational model we are teaching
OUTPUT_DIR = "./qwen-chat-finetune"    # Where the final trained adapter weights will be saved
DATA_JSON = "./qustions_and_quaries.json" # Your custom dataset containing NLP-to-SQL pairs

# --- Step 3: Memory Optimization (BitsAndBytes) ---
# A 7B model normally requires ~14GB+ VRAM. This 4-bit configuration shrinks it 
# so it can train on standard consumer GPUs (like RTX 3060/4060) without crashing.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                   # Compress the model to 4-bit precision
    bnb_4bit_quant_type="nf4",           # NormalFloat4: A special data type optimized for weights
    bnb_4bit_compute_dtype=torch.float16, # Math operations still happen in 16-bit for speed and accuracy
    bnb_4bit_use_double_quant=True       # Saves even more memory by quantizing the quantization constants
)

# --- Step 4: Tokenizer Setup ---
# The tokenizer converts human text into numbers the AI can understand.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Use the 'End of Sentence' token for padding blank spaces
tokenizer.padding_side = "right"          # Pad on the right side (standard for Causal Language Models)

# --- Step 5: Base Model Instantiation ---
# device_map="auto" automatically detects if the user has 1, 2, or 8 GPUs and splits the model safely.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", 
    trust_remote_code=True
)
# Prepares the model for 4-bit training by stabilizing certain mathematical layers
model = prepare_model_for_kbit_training(model)

# --- Step 6: LoRA Adapter Configuration (The "Brain Surgery") ---
# LoRA (Low-Rank Adaptation) freezes the base model and only trains a tiny attached neural network.
peft_config = LoraConfig(
    r=64,             # Rank: The size of the training adapter. Higher = smarter but uses more memory.
    lora_alpha=128,   # Scaling factor: Determines how strongly the new training overrides the base model.
    # Target modules: The specific mathematical layers inside the AI's "Attention" mechanism we want to train.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, # Randomly ignores 10% of neurons during training to prevent rote memorization (overfitting).
    bias="none",
    task_type="CAUSAL_LM" # We are training a text-generation (causal language) model.
)
# Inject the LoRA adapter into the base model
model = get_peft_model(model, peft_config)

# --- Step 7: Dataset Formatting & Tokenization ---
def format_data(example):
    """
    Formats the raw JSON data into the exact ChatML structure the Qwen model expects.
    It also masks the instruction text so the AI only learns to generate the SQL answer.
    """
    system_part = "<|im_start|>system\nYou are a Text-to-SQL expert. Answer only with SQL queries.<|im_end|>\n"
    instruction_part = f"<|im_start|>user\n{example['input']}<|im_end|>\n"
    response_part = f"<|im_start|>assistant\n{example['query']}<|im_end|>"
    
    full_prompt = system_part + instruction_part + response_part
    
    # Tokenize the entire prompt sequence. max_length=384 prevents long prompts from crashing the GPU.
    tokenized_full = tokenizer(full_prompt, truncation=True, max_length=384, padding=False)
    
    # CRITICAL STEP: Masking (-100)
    # PyTorch ignores labels marked as '-100' during loss calculation. 
    # We apply this to the system and user prompts so the AI is ONLY punished/rewarded for the SQL it generates.
    instruction_len = len(tokenizer(system_part + instruction_part)["input_ids"])
    labels = [-100] * instruction_len + list(tokenized_full["input_ids"])[instruction_len:]
    tokenized_full["labels"] = labels
    
    return tokenized_full

# Load the custom JSON dataset and apply the formatting function
dataset = load_dataset("json", data_files=DATA_JSON, split="train")
dataset = dataset.map(format_data, remove_columns=dataset.column_names)

# --- Step 8: Training Setup & Hyperparameters ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,  # How many examples to process at once per GPU. Keep low (2) for 4GB-8GB GPUs.
    gradient_accumulation_steps=4,  # Simulates a larger batch size (2 * 4 = 8) to stabilize learning without OOM.
    num_train_epochs=10,            # How many times the AI will review the entire dataset.
    learning_rate=2e-5,             # The speed of learning. 2e-5 is safe and prevents destroying base knowledge.
    lr_scheduler_type="cosine",     # Slowly decreases the learning rate at the end of training for precise tuning.
    warmup_ratio=0.1,               # Gradually increases learning rate for the first 10% of training to avoid shock.
    weight_decay=0.05,              # Penalizes overly complex weights to prevent overfitting.
    label_smoothing_factor=0.05,    # Prevents the AI from being "too confident", forcing it to learn actual logic.
    fp16=True,                      # Use 16-bit precision for faster training.
    logging_steps=1,                # Print training progress every step.
    save_strategy="epoch",          # Save a backup of the model after every epoch.
    optim="paged_adamw_32bit",      # Advanced optimizer that pages memory to CPU RAM if GPU VRAM gets too full.
    gradient_checkpointing=True,    # Trades compute time for memory savings (Crucial for small GPUs).
    report_to="none",               # Disables external logging platforms like Weights & Biases.
    gradient_checkpointing_kwargs={"use_reentrant": False} # Required for modern PyTorch versions.
)

# Initialize the huggingface Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    # DataCollator automatically pads batches of data to the same length dynamically.
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
)

# --- Step 9: Execution & Model Persistence ---
# Clear any residual GPU memory before starting the heavy training loop
torch.cuda.empty_cache()

# Start the training process!
trainer.train()

# Export final tuned adapter weights to the output directory
final_dir = f"{OUTPUT_DIR}/final_adapter"
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)

print(f"✅ Training Complete. Adapter weights safely exported to: {final_dir}")
