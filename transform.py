from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set paths
base_model_path = "meta-llama/Llama-3.2-1B"
adapter_path = "43e592e2-9234-587d-a3c2-00682c4ee8ba:train:0_sampler_weights_final" 
output_dir = "sft-no-olmo-tablegpt"


# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(base_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load and merge adapter
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

# Save merged model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

model_path = output_dir

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.chat_template = "{% for message in messages %}{{ message['role'].capitalize() }}: {{ message['content'] }}\n\n{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
tokenizer.save_pretrained(model_path)
print("Done:", tokenizer.chat_template[:80])

print("done")