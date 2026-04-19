from transformers import AutoTokenizer

model_path = "sft-math-balance"  # update this to your model path

# Created with the help of Copilot
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.chat_template = "{% for message in messages %}{{ message['role'].capitalize() }}: {{ message['content'] }}\n\n{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}"
tokenizer.save_pretrained(model_path)
print("Done:", tokenizer.chat_template[:80])
