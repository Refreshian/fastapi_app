import logging
import os
import re
import time
import pickle
import asyncio
import gc
import json
import traceback
import aiohttp
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic

from torch import bfloat16
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from search_data_elastic import elastic_query

import numpy as np
import redis.asyncio as redis
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Инициализация клиента Redis
redis_db = redis.Redis(host='localhost', port=6379, db=0)

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords')
nltk.download('punkt')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")

# Установка статуса GPU
async def set_gpu_status(status: str):
    logging.info(f"Устанавливается статус GPU: {status}")
    await redis_db.set("gpu:status", status)

# Сброс статуса GPU
async def reset_gpu_status():
    await set_gpu_status("idle")

# Загрузка словаря с темами
def load_dict_from_pickle(file_name):
    try:
        with open(file_name, 'rb') as f:
            your_dict = pickle.load(f)
        return your_dict
    except Exception as e:
        print(f"Произошла ошибка при загрузке файла: {e}")
        return None

async def generate_answers(client, prompt):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "Vikhr_Q3",
        "prompt": prompt,
        "stream": False
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                response_json = await response.json()
                return response_json.get("response", "")
            else:
                print(f"Ошибка при запросе к Ollama: {response.status}")
                return None

from ollama import AsyncClient

async def run_llm_query(task_data: dict):
    """Обрабатывает LLM-запрос с обновлением статуса задачи в Redis."""
    try:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Загружаем данные индекса
        file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
        indexes = load_dict_from_pickle(file_path)

        # Выполняем запрос к Elasticsearch
        data = []
        if task_data['query_str'] and task_data['query_str'] != 'all':
            search = task_data['query_str'].split(',')
            for query in search:
                data.extend(elastic_query(theme_index=indexes[int(task_data['index'])], query_str=query))
        else:
            data = elastic_query(theme_index=indexes[int(task_data['index'])], query_str='all', 
                                 min_date=task_data['min_date'], max_date=task_data['max_date'])

        # Получаем тексты и ограничиваем их количество
        texts = [x['text'] for x in data]
        texts = texts[:50000]  # Ограничение
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
        print('Всего текстов: {}'.format(total_texts)) 
        et = time.time()

        # Обновляем начальный статус задачи в Redis
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "in_progress",
            "total_texts": total_texts,
            "completed_texts": 0,
            "progress": 0,
        })

        llm_labels = []
        client = AsyncClient(host='http://localhost:11434')  # Создаем клиент один раз

        async def generate_answers(text, semaphore):
            async with semaphore:
                payload = {
                    "model": "Vikhr_Q3",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"У меня есть следующий текст:\n"
                                f"{text}\n\n"
                                "Есть ли в тексте фобии (страхи, предубеждения, опасения и т.д.) перед искусственным интеллектом (ИИ)? "
                                "Если есть - напиши кратко заголовок об этой фобии. Не пиши вводные в стиле 'В предоставленно тексте' и тп"
                                "Пиши только заголовок, напиши кратко в 1 предложение."
                                # "Если в сформированном заголовке нет фобии, то пиши 'Фобии перед ИИ нет'"
                            )
                        }
                    ]
                }

                if len(text) > 25000:
                    llm_labels.append("Длинный текст")
                    completed_texts = len(llm_labels)
                    progress = round((completed_texts / total_texts) * 100, 1)
                    await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                        "completed_texts": completed_texts,
                        "progress": progress
                    })
                    print("Длинный текст")
                    return

                with torch.no_grad():
                    response = await client.chat(model='Vikhr_Q3', messages=payload['messages'])
                    if response:
                        llm_labels.append(response['message']['content'])
                        completed_texts = len(llm_labels)
                        progress = round((completed_texts / total_texts) * 100, 1)
                        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                            "completed_texts": completed_texts,
                            "progress": progress
                        })
                    else:
                        llm_labels.append("bad response")
                        completed_texts = len(llm_labels)
                        progress = round((completed_texts / total_texts) * 100, 1)
                        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                            "completed_texts": completed_texts,
                            "progress": progress
                        })
                        print("bad response")

        async def main():
            semaphore = asyncio.Semaphore(10)
            tasks = [generate_answers(text, semaphore) for text in texts]
            await asyncio.gather(*tasks)

        await main()

        print(llm_labels[:5])
        elapsed_time = time.time() - et 
        print('Execution LLM time:', elapsed_time, 'seconds')

        # Обновляем статус задачи в Redis после завершения всех запросов
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "done", "completed_texts": total_texts, "progress": 100})
                
        llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

        gc.collect()
        torch.cuda.empty_cache()

        # Обработка эмбеддингов
        embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
        embeddings = []
        num_embeddings = len(llm_labels)

        # Обновляем статус перед началом процесса обработки эмбеддингов
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"embedding_status": "in_progress", "embedding_total": num_embeddings, "embedding_completed": 0, "embedding_progress": 0})

        # Генерация эмбеддингов с обновлением прогресса в Redis
        for i, label in enumerate(llm_labels):
            embedding = embedding_model.encode(label, show_progress_bar=False)
            embeddings.append(embedding)

            # Обновляем количество обработанных эмбеддингов и прогресс в Redis
            completed_embeddings = i + 1
            embedding_progress = round((completed_embeddings / num_embeddings) * 100, 1)
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={
                "embedding_completed": completed_embeddings,
                "embedding_progress": embedding_progress
            })

        if len(embeddings) > 0:
            # Преобразование списка эмбеддингов в массив NumPy
            embeddings = np.array(embeddings)

            umap_model = UMAP(n_neighbors=2, n_components=min(len(embeddings), 5), min_dist=0.0, metric="cosine", random_state=42)
            embeddings_umap = umap_model.fit_transform(embeddings)

            # Обновляем статус после завершения обработки эмбеддингов
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={"embedding_status": "done", "embedding_completed": num_embeddings, "embedding_progress": 100})
        else:
            print("Нет доступных эмбеддингов для обработки.")

        hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        hdbscan_model.fit(embeddings_umap)

        labels = hdbscan_model.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        num_noise_points = counts[np.where(unique_labels == -1)[0][0]] if -1 in unique_labels else 0

        # Используем преобразованные эмбеддинги для topic_model
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        topics, probs = topic_model.fit_transform(llm_labels, embeddings)  # Теперь `embeddings` - это NumPy массив

        async def generate_topic_label(client, key_words):
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": "llama3",
                "prompt": (
                    f"Используя ключевые слова: {key_words}, сгенерируй на русском языке короткий "
                    "(до 10 слов) и понятный заголовок для данной темы, пиши только сам заголовок на русском языке."
                ),
                "stream": False
            }
            with torch.no_grad():
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            return response_json.get("response", "")
                        else:
                            print(f"Ошибка при запросе к Ollama: {response.status}")
                            return None

        topic_labels_llama3 = []
        for i, topic in enumerate(topic_model.get_topics().values()):
            key_words = " | ".join(token[0] for token in topic[:10])
            label = await generate_topic_label(client, key_words)
            if label:
                topic_labels_llama3.append(label)

        for i, label in enumerate(topic_labels_llama3):
            print(f"Тема {i}: {label}")

        def shorten_by_words(text, max_words):
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'
            return text

        topic_labels_llama3 = [shorten_by_words(topic, 7) for topic in topic_labels_llama3]
        topic_model.set_topic_labels(topic_labels_llama3)

        fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                        hide_document_hover=False, custom_labels=True)

        file_location = f'/home/dev/fastapi/analytics_app/data/{task_data["user_id"]}/bertopic_files_directory/{task_data["folder_name"]}/'
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        new_filename = f"{indexes[int(task_data['index'])]}_{current_time}.html"
        fig.write_html(file_location + new_filename)

        filename = 'topic_model_' + new_filename.split('.html')[0]

        elapsed_time = time.time() - et
        total_seconds = int(elapsed_time)

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        execution_time = f"{hours} ч. {minutes} мин. {seconds} сек."
        print('Execution All time:', execution_time, 'seconds')

        try:
            os.chdir(file_location)
            topic_model.save(filename, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
            print(f"Модель успешно сохранена в: {file_location }")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")

        os.chdir(file_location)
        with open(f'my_list_llm_ans_{indexes[int(task_data["index"])]}_{current_time}.pkl', 'wb') as file:
            pickle.dump(llm_labels, file)

        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")

        file_info = {
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "model-file": filename,
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),
            "execution_llm_time": elapsed_time,
            "execution_all_time": execution_time,
            "query_str": task_data['query_str'], 
            "min_date": task_data['min_date'],
            "max_date": task_data['max_date'],
            "index_number": int(task_data['index']),
            "task_id": task_data['task_id'],
        }

        if user_data:
            if "bertopic_files_directory" in user_data:
                user_folders = json.loads(user_data["bertopic_files_directory"])
            else:
                user_folders = {}

            folder_name = task_data['folder_name']
            if folder_name in user_folders:
                user_folders[folder_name].append(file_info)
            else:
                user_folders[folder_name] = [file_info]

            serialized_folders = json.dumps(user_folders)
            await redis_db.hset(task_data["user_id"], "bertopic_files_directory", serialized_folders)
        else:
            raise Exception("User data does not exist.")

    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_data['task_id']}: {e}")
        traceback.print_exc()
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "failed", "error": str(e)})

    finally:
        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")
        print(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")

        async def reset_all_gpu_processes():
            import subprocess
            subprocess.call("nvidia-smi | awk '/[0-9]+/ {print $5}' | xargs -r kill -9", shell=True)

        if not torch.cuda.is_available():  
            await reset_all_gpu_processes()