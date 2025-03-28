from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import evaluate
import warnings as wr
import numpy as np
wr.filterwarnings('ignore')

# ✅ Load 10% of the dataset for initial runs
train_dataset = load_dataset('json', data_files='train.jsonl', split='train[:10%]')
eval_dataset = load_dataset('json', data_files='test.jsonl', split='train[:10%]')

# ✅ Mistral 7B Instruct Model and Tokenizer
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# ✅ Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

# ✅ Load tokenizer with special tokens
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# ✅ Reduce sequence length to 1024
max_length = 1024

# ✅ Prompt formatting function
def formatting_func(example):
    output_str = (
        f"Book Itself: {example['output']['book_itself']}\n"
        f"Categories: {example['output']['categories']}\n"
        f"Average Rating: {example['output']['average_rating']}\n"
        f"Ratings Count: {example['output']['ratings_count']}\n"
        f"Language: {example['output']['language']}\n"
        f"Similar Title Books: {', '.join(example['output']['similar_title_books'])}\n"
        f"Similar Description Books: {', '.join(example['output']['similar_description_books'])}\n"
        f"Similar Category Books FAISS: {', '.join(example['output']['similar_category_books_faiss'])}"
    )

    text = (
        f"### Instruction: {example['instruction']}\n"
        f"### Input: {example['input']}\n"
        f"### Output:\n{output_str}"
    )
    return text


# ✅ Tokenization and prompt generation
def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    result["labels"] = result["input_ids"].copy()
    return result

# ✅ Tokenize datasets
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

# ✅ LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'lm_head']
)

# ✅ Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# ✅ Check available GPUs and enable parallelism
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True

# ✅ Training Arguments (Optimized)
args = TrainingArguments(
    output_dir='mistral_7B_fine_tuned',
    num_train_epochs=1,  # ✅ Start with 1 epoch to test speed
    per_device_train_batch_size=4,  # ✅ Increased batch size
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # ✅ Effective batch size = 16
    warmup_ratio=0.03,
    optim='paged_adamw_8bit',
    logging_steps=10,
    save_strategy='steps',  # ✅ Save every 5000 steps
    save_steps=5000,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    bf16=True,  # ✅ Use BF16 for faster computation
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)

# ✅ Load Accuracy Metric
accuracy_metric = evaluate.load("accuracy")

# ✅ Compute Evaluation Metrics
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1).flatten()
    labels = labels.flatten()

    # ✅ Accuracy Calculation
    accuracy_results = accuracy_metric.compute(predictions=predictions, references=labels)

    # ✅ Perplexity Calculation
    eval_loss = torch.nn.CrossEntropyLoss()(torch.tensor(logits).permute(0, 2, 1), torch.tensor(labels)).item()
    perplexity = np.exp(eval_loss) if eval_loss < 100 else float('inf')

    return {
        "accuracy": accuracy_results['accuracy'],
        "perplexity": perplexity
    }

# ✅ Data Collator for Causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ✅ Initialize Trainer
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ✅ Start Fine-Tuning
model.config.use_cache = False
trainer.train()

# ✅ Save Final Model
trainer.save_model('mistral_7B_fine_tuned_final')

# ✅ Evaluate Model
eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)
print(f"Validation Results: {eval_results}")

# ✅ Merge and Save Merged Model
merged_model = model.merge_and_unload()
merged_model.save_pretrained('mistral_7B_merged')
tokenizer.save_pretrained('mistral_7B_merged')

# ✅ Push Merged Model to Hugging Face Hub
trainer.push_to_hub('sri-lasya/mistral-7B-book-recommendation')
