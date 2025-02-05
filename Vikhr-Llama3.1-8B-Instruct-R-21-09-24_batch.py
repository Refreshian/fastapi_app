import asyncio
import os
import io
import json
import pickle
import re
import tarfile
import time
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from torch import cuda
import aiohttp
from torch import bfloat16
import transformers
from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch
import nltk
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from search_data_elastic import elastic_query

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords')
nltk.download('punkt')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")

################################### data ###################################

# Загрузка словаря с темами
def load_dict_from_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            your_dict = pickle.load(f)
        return your_dict
    except Exception as e:
        print(f"Произошла ошибка при загрузке файла: {e}")
        return None

# Загружаем данные индекса
file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
indexes = load_dict_from_pickle(file_path)

index_el = 170
# Выполняем запрос к Elasticsearch
data = elastic_query(theme_index=indexes[index_el], query_str='all')

# Получаем тексты и ограничиваем их количество
texts = [x['text'] for x in data]
texts = texts[:10]  # Ограничение
total_texts = len(texts)

# Функция очистки текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление ссылок, символов и цифр
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Удаление стоп-слов
    text = ' '.join([word for word in text.split() if word not in russian_stopwords])
    return text

texts = [preprocess_text(x) for x in texts]
print('Всего текстов: {}'.format(total_texts))

# Проверьте доступность GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Используемое устройство: {device}')

# Исходные промты
system_prompt = """
<s>[INST] <<SYS>>
Ты дружелюбный ассистент для анализа и разметки текстов из социальных медиа. Отвечай только в указанном формате.
<</SYS>>
"""

async def generate_answers(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                # Сохраняем только поле "response"
                return response_json.get("response", "")
            else:
                print(f"Ошибка при запросе к Ollama: {response.status}")
                return None

async def main(data):
    tasks = []
    for text in data:
        user_prompt = (
            f"У меня есть следующий текст:\n"
            f"{text}\n\n"
            "Есть ли в тексте фобии (страхи, предубеждения, опасения и т.д.) перед искусственным интеллектом (ИИ)? "
            "Если они есть - в чем причина фобии? Отвечай кратко (до 2 предложений)"
        )
        task = asyncio.create_task(generate_answers(user_prompt))
        tasks.append(task)

    llm_labels = await asyncio.gather(*tasks)
    return llm_labels

st = time.time()
# Запускаем основную асинхронную функцию
results = asyncio.run(main(texts))

for content in results:
    if content:
        print(content)

# Заканчиваем подсчет времени
elapsed_time = time.time() - st 
print('Execution time:', elapsed_time, 'seconds')