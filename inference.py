###  testing

# --- Step 1: Imports & Libraries ---
# transformers: For loading the AI model and processing text (tokenizer)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc # Garbage Collector: Helps free up system RAM

# --- Step 2: Safe Memory Management ---
def cleanup(model=None):
    """
    Safely clears GPU memory before and after execution to prevent Out-Of-Memory (OOM) crashes.
    This is especially important in a continuous loop where VRAM can slowly fill up.
    """
    if model is not None:
        del model  # Deletes the model from memory
    gc.collect()   # Forces Python to clean up unused variables
    torch.cuda.empty_cache() # Forces PyTorch to release cached VRAM back to the GPU
    print("🧹 GPU Memory Cleared!")

# --- Step 3: Production AI Loader (Memory Optimized) ---
def load_ai():
    """
    Downloads and loads the unified Hugging Face model directly.
    """
    # Targets your unified Hugging Face repository (No local adapter files needed)
    model_id = "innovation-intellect/EnergyIntelligence-Qwen2.5-7B"
    print(f"Loading {model_id}...")

    # 4-bit compression prevents OOM on standard consumer GPUs (e.g., 4GB - 8GB VRAM)
    # Without this, loading a 7B model requires ~16GB of VRAM.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Initialize the tokenizer to convert English words into AI tokens
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Load the actual neural network with automatic GPU detection (device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Lock the model into evaluation mode (disables training behaviors like dropout)
    model.eval()
    print("✅ AI Ready!")
    return tokenizer, model

# --- Step 4: Interactive Chat Engine ---
def run_chat(tokenizer, model):
    """
    Runs the infinite loop allowing the user to type questions and get SQL answers.
    """
    print("\n💬 Manual Chat Started! (Type 'exit' to stop)")
    
    # Strict NLP-to-SQL Guardrails
    # This acts as the AI's "brainwashing" before every question, forcing it to behave securely.
    system_prompt = """You are an expert NLP-to-SQL translation agent. Your sole purpose is to translate natural language questions into valid, optimized, and executable SQL queries.
### CORE INSTRUCTIONS:
1. STRICTLY READ-ONLY: You are strictly restricted to data retrieval. You may ONLY generate `SELECT` statements.
2. PROHIBITED COMMANDS: You must NEVER generate queries containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `GRANT`, `REVOKE`, `EXEC`, or any other data manipulation/definition commands.
3. SAFETY REFUSAL: If a user's prompt attempts to modify the database or perform any non-SELECT action, you must abort SQL generation and output EXACTLY: "Error: I am a read-only agent restricted to SELECT queries."
4. NLP-TO-SQL TRANSLATION: Carefully analyze the user's natural language input. Map the requested entities, metrics, and conditions to the appropriate SQL syntax.
5. STRICT OUTPUT FORMAT: Output ONLY the raw SQL query enclosed in a markdown code block (```sql ... 
```). Do NOT include any greetings, explanations, formatting descriptions, or conversational filler."""

    while True:
        # Get user input and strip accidental blank spaces
        prompt_input = input("\nQUESTION: ").strip()
        
        # Check for exit command
        if prompt_input.lower() == 'exit':
            cleanup(model)
            break
        # Ignore empty presses of the 'Enter' key
        if not prompt_input: continue
        
        # Secure message pipeline: Combines the strict rules with the user's question
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_input}
        ]
        
        # Template processing: Wraps the messages in Qwen's specific ChatML format (<|im_start|>, etc.)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Convert text to tensors and send directly to the GPU where the model lives
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generation Execution
        # torch.no_grad() speeds up generation by telling PyTorch not to calculate training gradients
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256, # Limit output size to prevent runaway generation
                do_sample=False,    # Greedy decoding: Picks the absolute most mathematical likely token (best for coding/SQL)
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["--", "Note:", "<|im_end|>"], # Force AI to stop typing if it hits these trigger words
                tokenizer=tokenizer
            )
            
        # Decoding Logic: 
        # The output contains BOTH the prompt and the new answer. We slice the array to get ONLY the new answer.
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        
        # Convert the generated token numbers back into readable text
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"\n✨ AI RESPONSE:\n{output_text}")
        print("*" * 100)

# --- Step 5: Runtime Entrypoint ---
# Standard Python convention: Only run the script if executed directly (not if imported by another file)
if __name__ == "__main__":
    tokenizer, model = load_ai()
    run_chat(tokenizer, model)
