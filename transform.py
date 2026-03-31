from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set your paths
base_model_path = "meta-llama/Llama-3.2-1B"  # e.g., "meta-llama/Llama-2-7b-hf"
# Use the actual adapter folder path on disk; avoid Hugging Face repo-style path naming here.
adapter_path = "6bec8f14-1a73-52f0-8f3e-fc7d981825f1:train:0_sampler_weights_final"
output_dir = "merged_model_output_test"

# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load and merge adapter
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

# Save merged model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Merged model saved to {output_dir}")