import yake
from transformers import AutoTokenizer, AutoModel
import torch


# Модель для генерации эмбеддингов
model_name = "intfloat/e5-large-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().numpy()


#Функция для извлечения ключевых слов с использованием Yape
def extract_keywords(text):
    extractor = yake.KeywordExtractor(
        lan="en",
        top=4
    )
    keywords = extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords]