import os
import json
import pickle
import re
import time
import gc
import torch
import nltk
from torch import cuda, bfloat16
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # Импорт vLLM

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords')
nltk.download('punkt')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")

# Устанавливаем окружение
os.environ["SUNO_USE_SMALL_MODELS"] = "True"  # Установка для оптимизации памяти

# Убедитесь, что CUDA доступна
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

from search_data_elastic import elastic_query

model_name = "Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(f"/home/dev/fastapi/analytics_app/data/LLM_models/{model_name}")
# Инициализация vLLM
vllm_model = LLM(model=f"/home/dev/fastapi/analytics_app/data/LLM_models/{model_name}")

# загрузка словаря с темами
def load_dict_from_pickle(file_name):
    """
    Загружает словарь из файла Pickle.
    :param file_name: Имя файла (str), из которого нужно загрузить словарь.
    :return: Загруженный словарь (dict) или None, если загрузка не удалась.
    """
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

index_el = 145
# Выполняем запрос к Elasticsearch
data = elastic_query(theme_index=indexes[index_el], query_str='all')

# Получаем тексты и ограничиваем их количество
texts = [x['text'] for x in data]
texts = texts[:300]  # Ограничение
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

# Предобработка текстов
texts = [preprocess_text(x) for x in texts]


# Конфигурируем vLLM
# vllm_config = LLMConfig(
#     model_path=f"/home/dev/fastapi/analytics_app/data/LLM_models/{model_name}",
#     tokenizer=tokenizer,
#     max_new_tokens=50,
#     temperature=0.6,
#     repetition_penalty=1.1,
# )

# sampling_params = SamplingParams(model_name=model_name)

# Параметры батча
batch_size = 5
llm_labels = []

# Начало таймера
start_time = time.time()

# Обработка текстов батчами
for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]  # Получаем подмассив текстов для батча

    # Формируем сообщения для vLLM
    messages = [
        f"<s>[INST] <<SYS>> Ты дружелюбный ассистент для анализа и разметки текстов из социальных медиа. <</SYS>> У меня есть следующий текст:\n{text}\nНа основе текста выпиши ключевые слова и составь краткий заголовок. Обрати внимание на то, что необходимо написать только заголовок." 
        for text in batch_texts
    ]

    # Генерация ответов
    with torch.no_grad():
        responses = vllm_model.generate(messages, SamplingParams(temperature=0.8, max_tokens=100))

        print(responses[0])
        print(555777)
        print(responses[0].outputs[0].text)

        for res in responses:
            if "generated_text" in res:
                llm_labels.append(res['generated_text'])

# Тайм о всем процессе
elapsed_time = time.time() - start_time
print('Execution time:', elapsed_time, 'seconds')

# Дополнительный вывод или сохранение результатов
print("Generated Labels:", llm_labels)