from transformers import AutoTokenizer, LlamaForCausalLM

model_path = "/Users/fengyuan/Documents/models/"
model = LlamaForCausalLM.from_pretrained(model_path + "Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained(model_path + "Llama-2-7b-hf")

prompt = "Simply put, the theory of relativity states that "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=1024)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
