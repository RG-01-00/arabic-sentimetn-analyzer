from transformers import AutoTokenizer

MODEL_NAME = "aubmindlab/bert-base-arabertv02"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained("./models/best_bert_model")

print("Tokenizer saved successfully.")