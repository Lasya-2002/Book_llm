from transformers import pipeline,AutoModelForCausalLM,AutoTokenizer

# ✅ Load merged model and tokenizer
model_path = 'mistral_24B_merged'
tokenizer_path = 'mistral_24B_merged'

# ✅ Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# ✅ Define inference pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# ✅ Generate recommendations
prompt = (
    "### Instruction: Recommend books based on a preferred genre\n"
    "### Input: I enjoy reading books in the 'Science Fiction' genre. Can you suggest similar books?\n"
    "### Output:"
)
results = generator(prompt, max_length=512, num_return_sequences=1, temperature=0.7)

print("Generated Output:\n", results[0]['generated_text'])
