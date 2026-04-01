from transformers import AutoTokenizer

instruct_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
merged_tok = AutoTokenizer.from_pretrained("path/sft-4")

merged_tok.chat_template = instruct_tok.chat_template
merged_tok.save_pretrained("path/sft-4")
print("Done:", merged_tok.chat_template[:80])