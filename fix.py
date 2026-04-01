from transformers import AutoTokenizer

instruct_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
merged_tok = AutoTokenizer.from_pretrained("/users/PAS2526/carterglazer/glazer77/testFinal/merged_model_output_test")

merged_tok.chat_template = instruct_tok.chat_template
merged_tok.save_pretrained("/users/PAS2526/carterglazer/glazer77/testFinal/merged_model_output_test")
print("Done:", merged_tok.chat_template[:80])