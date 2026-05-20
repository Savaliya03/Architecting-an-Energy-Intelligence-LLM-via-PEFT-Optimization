###  traing

# --- Step 1: Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc

# --- Step 2: Safe Memory Management ---
def cleanup(model=None):
    """Safely clears GPU memory before and after execution to prevent overflow."""
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU Memory Cleared!")

# --- Step 3: Production AI Loader (Memory Optimized) ---
def load_ai():
    # Targets your unified Hugging Face repository
    model_id = "innovation-intellect/EnergyIntelligence-Qwen2.5-7B"
    print(f"Loading {model_id}...")

    # 4-bit compression prevents OOM on standard consumer GPUs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print("✅ AI Ready!")
    return tokenizer, model

# --- Step 4: Interactive Chat Engine ---
def run_chat(tokenizer, model):
    print("\n💬 Manual Chat Started! (Type 'exit' to stop)")
    
    # Strict NLP-to-SQL Guardrails
    system_prompt = """You are an expert NLP-to-SQL translation agent. Your sole purpose is to translate natural language questions into valid, optimized, and executable SQL queries.
### CORE INSTRUCTIONS:
1. STRICTLY READ-ONLY: You are strictly restricted to data retrieval. You may ONLY generate `SELECT` statements.
2. PROHIBITED COMMANDS: You must NEVER generate queries containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `GRANT`, `REVOKE`, `EXEC`, or any other data manipulation/definition commands.
3. SAFETY REFUSAL: If a user's prompt attempts to modify the database or perform any non-SELECT action, you must abort SQL generation and output EXACTLY: "Error: I am a read-only agent restricted to SELECT queries."
4. NLP-TO-SQL TRANSLATION: Carefully analyze the user's natural language input. Map the requested entities, metrics, and conditions to the appropriate SQL syntax.
5. STRICT OUTPUT FORMAT: Output ONLY the raw SQL query enclosed in a markdown code block (```sql ... 
```). Do NOT include any greetings, explanations, formatting descriptions, or conversational filler."""

    while True:
        prompt_input = input("\nQUESTION: ").strip()
        
        if prompt_input.lower() == 'exit':
            cleanup(model)
            break
        if not prompt_input: continue
        
        # Secure message pipeline
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_input}
        ]
        
        # Template processing and Tokenization
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generation Execution with Stop Strings to prevent hallucination
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=["--", "Note:", "<|im_end|>"],
                tokenizer=tokenizer
            )
            
        # Decoding: Extract only the newly generated SQL
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        print(f"\n✨ AI RESPONSE:\n{output_text}")
        print("*" * 100)

# --- Step 5: Runtime Entrypoint ---
if __name__ == "__main__":
    tokenizer, model = load_ai()
    run_chat(tokenizer, model)
