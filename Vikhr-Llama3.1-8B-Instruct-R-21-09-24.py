import asyncio
import os, io, json
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
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
# os.environ["SUNO_USE_SMALL_MODELS"] = "True"

from torch import bfloat16
import transformers

from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch, os, json
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from search_data_elastic import elastic_query
import nltk

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords')
nltk.download('punkt')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")


################################### data ###################################

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

index_el = 170
# Выполняем запрос к Elasticsearch
data = elastic_query(theme_index=indexes[index_el], query_str='all')

# # Фильтруем данные по дате
# task_data['min_date'] = int(task_data['min_date'])
# task_data['max_date'] = int(task_data['max_date'])
# data = [x for x in data if task_data['min_date'] <= x['timeCreate'] <= task_data['max_date']]

# Получаем тексты и ограничиваем их количество
texts = [x['text'] for x in data]
texts = texts[:20]  # Ограничение
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

llm_labels = []
count = 0
et = time.time()  # Начало отсчета времени

url = "http://localhost:8000/v1/chat/completions"
model_path = "/home/dev/fastapi/analytics_app/data/LLM_models/Quant/Vikhrmodels-Vikhr-Llama3.1-8B-Instruct-R-21-09-24-Q3_K_M.gguf"

system_prompt = """
<s>[INST] <<SYS>>
Ты дружелюбный ассистент для анализа и разметки текстов из социальных медиа. Отвечай только в указанном формате.
<</SYS>>
"""

async def fetch(session, text, semaphore):
    async with semaphore:
        user_message = {
            "role": "user",
            "content": (
                f"{system_prompt}\n"
                f"У меня есть следующий текст:\n"
                f"{text}\n\n"
                "Есть ли в тексте фобии (страхи, предубеждения, опасения и т.д.) перед искусственным интеллектом (ИИ)? Если они есть - в чем причина фобии? Отвечай кратко (до 2 предложений)"
            )
        }
        
        data = {
            "model": model_path,
            "messages": [user_message]
        }
        
        try:
            async with session.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data), timeout=60) as response:
                response.raise_for_status()  # Raise an error for bad responses
                result = await response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    # Сохранение ответа в нужном формате
                    return result['choices'][0]['message']['content']
                else:
                    print(f"Unexpected response structure: {result}")
                    return None
        except asyncio.TimeoutError:
            print(f"Timeout error for text: {text}")
            return None
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

async def main(texts):
    semaphore = asyncio.Semaphore(5)  # Limit concurrent requests to 10
    llm_labels = []  # Массив для хранения ответов модели
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, text, semaphore) for text in texts]
        results = await asyncio.gather(*tasks)
        # Сохраняем все полученные результаты в массив llm_labels
        llm_labels.extend(result for result in results if result is not None)
    return llm_labels

# Пример вызова функции main
llm_labels = asyncio.run(main(texts))

for content in llm_labels:
    if content:
        print(content)


# Заканчиваем подсчет времени
st = time.time()
elapsed_time = st - et
print('Execution time:', elapsed_time, 'seconds')


# ################################### BERTopic ###################################

# from sentence_transformers import SentenceTransformer
# from umap import UMAP
# from hdbscan import HDBSCAN
# from sklearn.metrics import silhouette_score
# import numpy as np
# import re
# from bertopic import BERTopic
# gc.collect()
# torch.cuda.empty_cache()

# # Шаг 1 - Очистка данных
# llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

# # Шаг 2 - Генерация эмбедингов
# embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
# embeddings = embedding_model.encode(llm_labels, show_progress_bar=True)

# # Шаг 3 - Снижение размерности UMAP
# # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
# # embeddings_umap = umap_model.fit_transform(embeddings)

# # Проверим, что embeddings не пуст или
# if len(embeddings) > 0:
#     umap_model = UMAP(n_neighbors=2, n_components=min(len(embeddings), 5), min_dist=0.0, metric="cosine", random_state=42)
#     embeddings_umap = umap_model.fit_transform(embeddings)
# else:
#     print("Нет доступных эмбеддингов для обработки.")

# # Шаг 4 - Кластеризация HDBSCAN
# hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
# hdbscan_model.fit(embeddings_umap)

# # Вывод результатов кластеризации
# labels = hdbscan_model.labels_

# # Подсчет уникальных значений и их количества
# unique_labels, counts = np.unique(labels, return_counts=True)

# # Число кластеров (кроме шума `-1`)
# num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

# # Число шумовых точек
# if -1 in unique_labels:
#     noise_index = np.where(unique_labels == -1)[0][0]  # Найти индекс метки `-1` в unique_labels
#     num_noise_points = counts[noise_index]
# else:
#     num_noise_points = 0
# # Шаг 5 - Тематическое моделирование BERTopic
# topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
# topics, probs = topic_model.fit_transform(llm_labels, embeddings)

# # Шаг 6 - Генерация заголовков тем
# topic_labels = topic_model.generate_topic_labels(nr_words=10)  # Например, по 3 ключевых слова на тему
# for i, label in enumerate(topic_labels):
#     print(f"Тема {i}: {label}")

# pipe = pipeline(
#     model=model,
#     tokenizer=tokenizer,
#     task='text-generation',
#     temperature=0.6,
#     max_new_tokens=50,
#     repetition_penalty=1.1,
# )
# # pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

# # Шаг 6 - Генерация заголовков тем
# topic_labels_llama3 = []

# for i, topic in enumerate(topic_model.get_topics().values()):  # Получаем ключевые слова тем
#     key_words = " | ".join(token[0] for token in topic[:10])  # Берем 5 ключевых слов для темы

#     # Формируем сообщения
#     messages = [
#         {"role": "system", "content": f"[INST] Используя данные ключевые слова: {key_words}, сгенерируй на русском языке короткий (не более 5-7 слов) и понятный заголовок для данной темы. Не пиши какие ключевые слова ты использовал (Using keywords), не пиши дополнительных пояснений для заголовка, пиши только сам заголовок на русском языке. [/INST]"}
#     ]
    
#     # Очищаем кэш перед вызовом модели
#     torch.cuda.empty_cache()
    
#     # Используем torch.no_grad() для предотвращения вычисления градиентов
#     with torch.no_grad():
#         response = pipe(messages, num_return_sequences=1)
    
#     # Обрабатываем ответ
#     topic_labels_llama3.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', ''))

#     # generated_label = response[0]['generated_text'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', '')
#     # topic_labels_llama3.append(f"Тема {i}: {generated_label}")

# for i, label in enumerate(topic_labels_llama3):
#     print(f"Тема {i}: {label}")


# print(topic_labels_llama3)

# # Шаг 7 - Визуализации
# topic_model.visualize_topics()
# # topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap)

# # ################################### Visualize ###################################

# fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
#                                 hide_document_hover=False, custom_labels=True)

# # Модифицируйте метки
# # for trace in fig.data:
# #     trace.name = ' '.join(trace.name.split()[:10])  # Оставляем только первые 3 слова в метке

# filename = 'kibersport_01.01.2024-31.12.2024.json'
# # print(filename)
# os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
# fig.write_html(filename.split('.json')[0] + '_' + model_name + '.html')

# ###################################### save model #################################

# # from pathlib import Path
# # from PIL import Image
# # import joblib  # или import pickle
# # # Задайте директорию для сохранения файлов
# # filename = 'topic_model_' + filename.split('.json')[0]
# # save_directory = Path("/home/dev/fastapi/analytics_app/data/html_files")  # укажите путь к директории
# # topics_file_path = save_directory / filename
# # print(topics_file_path)

# # # Проверяем, существует ли файл и удаляем его
# # if topics_file_path.exists():
# #     os.remove(topics_file_path)
# #     print(f"Удален старый файл: {topics_file_path}")

# # # Теперь сохраняем темы
# # try:
# #     # joblib.dump(model, 'bertopic_model.joblib')
# #     os.chdir(save_directory)
# #     topic_model.save(filename, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)


# #     print(f"Модель успешно сохранена в: {save_directory / filename }")
# # except Exception as e:
# #     print(f"Ошибка при сохранении модели: {e}")


# # os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
# # # Сохранение списка в файл с помощью pickle
# # with open('my_list_llm_ans.pkl', 'wb') as file:
# #     pickle.dump(llm_labels, file)


# print('Длинных текстов:', count)
# print(f'llm_labels: {len(llm_labels)}')