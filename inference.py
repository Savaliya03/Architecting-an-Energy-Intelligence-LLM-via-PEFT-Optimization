from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

def cleanup():
    global model
    if 'model' in globals(): del model
    gc.collect()
    torch.cuda.empty_cache()

# Load the merged model directly from Hugging Face
model_id = "innovation-intellect/EnergyIntelligence-Qwen2.5-7B"
print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("AI is ready now!")

def run_chat():
    print("\n💬 Manual Chat Started! (Type 'exit' to stop)")
    while True:
        prompt_input = input("\nQUESTION: ").strip()
        if prompt_input.lower() == 'exit':
            cleanup()
            break
        if not prompt_input: continue

        messages = [
            {
                "role": "system",
                "content": "You are an expert NLP-to-SQL translation agent. STRICTLY READ-ONLY. Generate only SELECT statements. Output ONLY the raw SQL query enclosed in a markdown code block."
            }
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"\n✨ AI RESPONSE:\n{output_text}")
        print("*" * 100)

if __name__ == "__main__":
    run_chat()