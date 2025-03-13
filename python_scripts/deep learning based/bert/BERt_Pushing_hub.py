from transformers import BertForSequenceClassification, BertTokenizer
from huggingface_hub import upload_folder,HfApi

model = BertForSequenceClassification.from_pretrained("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/python_scripts/bert_finetuned_books")
tokenizer = BertTokenizer.from_pretrained("C:/Users/mvsla/OneDrive/Documents/GitHub/Book_llm/python_scripts/bert_finetuned_books")

api = HfApi()
api.create_repo("sri-lasya/book-bert")


# Push model and tokenizer
model.push_to_hub("sri-lasya/book-recommender-bert")
tokenizer.push_to_hub("sri-lasya/book-recommender-bert")

