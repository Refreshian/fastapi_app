import pickle
import re
import gc
import time
from tqdm import tqdm
import numpy as np
import torch
from tritonclient import grpc as triton_grpc
from tritonclient.utils import triton_to_np_dtype

# Возможно, вам нужно изменить местоположение импорта InferInput.
try:
    from tritonclient.grpc import InferInput
except ImportError as e:
    print(f"Ошибка импорта InferInput: {e}")
    # Если не удалось импортировать, вы можете предпринять шаги, чтобы сообщить об этом.

from search_data_elastic import elastic_query
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")

# Загружаем словарь с темами
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

index_el = 145
# Выполняем запрос к Elasticsearch
data = elastic_query(theme_index=indexes[index_el], query_str='all')

# Получаем тексты и ограничиваем их количество
texts = [x['text'] for x in data][:1000] 
total_texts = len(texts)

# Функция очистки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in russian_stopwords])
    return text

texts = [preprocess_text(x) for x in texts]
print(f'Всего текстов: {total_texts}')

# Подключение к Triton
triton_url = "localhost:8001"  # Укажите ваш URL сервера Triton
model_name = "Meta-Llama-3-8B-Instruct"

# Определяем клиент Triton
triton_client = triton_grpc.InferenceServerClient(url=triton_url)

# Исходные промты
system_prompt = """
<s>[INST] <<SYS>>
Ты дружелюбный ассистент для анализа и разметки текстов из социальных медиа. Отвечай только в указанном формате.
<</SYS>>
"""

batch_size = 5
llm_labels = []

et = time.time()  # Начало отсчета времени

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]  # Получаем подмассив текстов для батча

    # Создаем входные данные для Triton
    input_texts = [f"{system_prompt}У меня есть следующий текст:\n{text}\nНа основе текста выпиши ключевые слова (до 9 слов) и составь краткий заголовок" for text in batch_texts]

    # Кодируем строки в байты с помощью utf-8
    encoded_texts = [text.encode('utf-8') for text in input_texts]

    # Создайте массив текстовых данных с правильным типом
    inputs_np = np.array(encoded_texts, dtype=np.bytes_)

    # Создайте объект InferInput с указанием типа данных
    inputs_dict = InferInput("input_text", inputs_np.shape, "BYTES")

    # Выполняем запрос к модели в Triton
    try:
        response = triton_client.infer(model_name=model_name, inputs=[inputs_dict], outputs=None)
    except Exception as e:
        print(f"Ошибка при выполнении запроса к модели: {e}")
        continue

    # Обработка ответа
    outputs = response.get_response()  # Получаем ответ
    llm_labels.extend(outputs['outputs'])  # Добавьте ваш способ извлечения результата

    # Следим за длинными текстами
    for text in batch_texts:
        if len(text) > 15000:
            llm_labels.append('Длинный текст')

# Заканчиваем подсчет времени
st = time.time()
elapsed_time = st - et
print('Execution time:', elapsed_time, 'seconds')