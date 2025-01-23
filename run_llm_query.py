import logging
import os
import re
import time
import pickle
import asyncio
import gc, json
import traceback
# import redis
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
import transformers
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from search_data_elastic import elastic_query

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import redis.asyncio as redis
# Инициализация клиента Redis
redis_db = redis.Redis(host='localhost', port=6379, db=0)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

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
            data = elastic_query(theme_index=indexes[int(task_data['index'])], query_str='all')

        # Фильтруем данные по дате
        task_data['min_date'] = int(task_data['min_date'])
        task_data['max_date'] = int(task_data['max_date'])
        data = [x for x in data if task_data['min_date'] <= x['timeCreate'] <= task_data['max_date']]

        # Получаем тексты и ограничиваем их количество
        et = time.time()
        texts = [x['text'] for x in data]
        texts = texts[:2000]  # Ограничение
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

        model_name = 'Vikhr-Llama3.1-8B-Instruct-R-21-09-24'

        # Обновляем начальный статус задачи в Redis
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={
            "status": "in_progress",
            "total_texts": total_texts,
            "completed_texts": 0,
            "progress": 0,
        })

        ################################### Модель ###################################

        tokenizer = AutoTokenizer.from_pretrained(
            f"/home/dev/fastapi/analytics_app/data/LLM_models/{model_name}"
        )

        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit квантование
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Вторичное квантование
            bnb_4bit_compute_dtype=bfloat16  # Тип вычислений
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            f"/home/dev/fastapi/analytics_app/data/LLM_models/{model_name}",
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map='auto',
        )

        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1,
        )

        ################################### Промты ###################################

        system_prompt = """
        <s>[INST] <<SYS>>
        Ты дружелюбный ассистент для анализа текстов из социальных медиа. Отвечай только в указанном формате.
        <</SYS>>
        """
        batch_size = 20
        llm_labels = []
        processed_texts = {}
        et = time.time()  # Начало отсчета времени
        gc.collect()
        torch.cuda.empty_cache()
        count_long_texts = 0
        completed_texts = 0

        # Генерация заголовков
        # Проходим через массив текстов, обрабатывая их батчами
        for i in tqdm(range(0, len(texts), batch_size)):
            await asyncio.sleep(0.01)
            
            batch_texts = texts[i:i + batch_size]  # Получаем подмассив текстов для батча

            # Создаем сообщения для LLM, добавляя ожидаемый формат ответа
            messages = [
                    # {
                    #     "role": "system", 
                    #     "content": (
                    #         f"{system_prompt}"
                    #         f"Вот текст:\n"
                    #         f"{text}\n\n"
                    #         "Согласно этому тексту, выдели ключевые слова и сформулируй заголовок. "
                    #         "Пожалуйста, сообщи только заголовок."
                    #     )
                    # }

                    {
                        "role": "system", 
                        "content": (
                            f"{system_prompt}"
                            f"У меня есть следующий текст:\n"
                            f"{text}\n\n"
                            # "На основе текста выпиши ключевые слова (до 9 слов) и составь краткий заголовок"
                            "На основе текста составь краткий заголовок"

                        )
                    }
                for text in batch_texts
                if len(text) <= 15000  # Игнорируем длинные тексты сразу в исходном массиве
            ]
            
            # Проверяем, есть ли тексты для обработки в данной партии
            if not messages:
                completed_texts += len(batch_texts)
                progress = round((completed_texts / total_texts) * 100, 1)
                
                # Записываем прогресс в Redis
                await redis_db.hset(f"task:{task_data['task_id']}", mapping={"completed_texts": completed_texts, "progress": progress})
                continue  # Переходим к следующему батчу

            # Очищаем кэш перед вызовом модели
            torch.cuda.empty_cache()
            
            # Используем torch.no_grad() для предотвращения вычисления градиентов
            with torch.no_grad():
                # Генерируем ответ для батча, делая вызов вашей модели
                response = pipe(messages, num_return_sequences=1)

                # Итерируем по результатам
                for res in response:
                    try:
                        # Проверяем, что res является словарем и содержит нужный ключ
                        if isinstance(res, dict) and 'generated_text' in res:
                            # Добавляем содержимое сгенерированного текста без удаления
                            for text_result in res['generated_text']:
                                if isinstance(text_result, dict) and 'content' in text_result:
                                    # Извлекаем текст, убирая ненужные символы
                                    generated_content = text_result['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', '')
                                    llm_labels.append(generated_content)
                                else:
                                    print("Unexpected generated text format:", text_result)
                        else:
                            print("Unexpected result format:", res)
                    except Exception as e:
                        print("Error processing result:", e)

            # Следим за длинными текстами
            for text in batch_texts:
                if len(text) > 15000:
                    llm_labels.append('Длинный текст')
                    count_long_texts += 1

            # Обновляем прогресс после обработки каждого текста
            completed_texts += len(batch_texts)
            progress = round((completed_texts / total_texts) * 100, 1)
            
            # Записываем прогресс в Redis
            await redis_db.hset(f"task:{task_data['task_id']}", mapping={"completed_texts": completed_texts, "progress": progress})

        # Финальное обновление статуса задачи
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "done", "completed_texts": total_texts, "progress": 100})

        ################################### BERTopic ###################################

        # Шаг 1 - Очистка данных
        llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

        gc.collect()
        torch.cuda.empty_cache()

        # Шаг 2 - Генерация эмбедингов
        embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
        embeddings = embedding_model.encode(llm_labels, show_progress_bar=True)

        # Шаг 3 - Снижение размерности UMAP
        if len(embeddings) > 0:
            umap_model = UMAP(n_neighbors=2, n_components=min(len(embeddings), 5), min_dist=0.0, metric="cosine", random_state=42)
            embeddings_umap = umap_model.fit_transform(embeddings)
        else:
            print("Нет доступных эмбеддингов для обработки.")
        
        # Шаг 4 - Кластеризация HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        hdbscan_model.fit(embeddings_umap)

        # Вывод результатов кластеризации
        labels = hdbscan_model.labels_

        # Подсчет уникальных значений и их количества
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Число кластеров (кроме шума `-1`)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Число шумовых точек
        if -1 in unique_labels:
            noise_index = np.where(unique_labels == -1)[0][0]  # Найти индекс метки `-1` в unique_labels
            num_noise_points = counts[noise_index]
        else:
            num_noise_points = 0

        # Шаг 5 - Тематическое моделирование BERTopic
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        topics, probs = topic_model.fit_transform(llm_labels, embeddings)

        # Шаг 6 - Генерация заголовков тем
        topic_labels = topic_model.generate_topic_labels(nr_words=10)  # Например, по 6 ключевых слова на тему
        for i, label in enumerate(topic_labels):
            print(f"Тема {i}: {label}")

        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1,
        )

        # Шаг 6 - Генерация заголовков тем
        topic_labels_llama3 = []

        for i, topic in enumerate(topic_model.get_topics().values()):  # Получаем ключевые слова тем
            key_words = " | ".join(token[0] for token in topic[:10])  # Берем 10 ключевых слов для темы

            # Формируем сообщения
            messages = [
                {"role": "system", "content": f"[INST] Используя ключевые слова: {key_words}, сгенерируй на русском языке короткий (до 8 слов) и понятный заголовок для данной темы, пиши только сам заголовок на русском языке. [/INST]"}
            ]
            
            # Очищаем кэш перед вызовом модели
            torch.cuda.empty_cache()
            
            # Используем torch.no_grad() для предотвращения вычисления градиентов
            with torch.no_grad():
                response = pipe(messages, num_return_sequences=1)
            
            # Обрабатываем ответ
            topic_labels_llama3.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', ''))

        for i, label in enumerate(topic_labels_llama3):
            print(f"Тема {i}: {label}")


        def shorten_by_words(text, max_words):
            """Сокращает текст до заданного количества слов с добавлением многоточия."""
            words = text.split()  # Разделяем текст на слова
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'  # Сокращаем и добавляем многоточие
            return text  # Если длина не превышает, возвращаем оригинальный текст

        # Сокращение всех тем до 7 слов
        topic_labels_llama3 = [shorten_by_words(topic, 7) for topic in topic_labels_llama3]

        topic_model.set_topic_labels(topic_labels_llama3)
        
        # Визуализация
        fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                        hide_document_hover=False, custom_labels=True)

        # Устанавливаем путь к директории файла
        file_location = f'/home/dev/fastapi/analytics_app/data/{task_data['user_id']}/bertopic_files_directory/{task_data['folder_name']}/'

        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        # Формируем новое имя файла с добавлением даты и времени
        new_filename = f"{indexes[int(task_data['index'])]}_{current_time}.html"
        fig.write_html(file_location + new_filename)

          
        # Записываем прогресс в Redis
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"completed_texts": completed_texts*batch_size, "progress": progress})

        print(f'llm_labels: {len(llm_labels)}')

        # Финальное обновление статуса задачи
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "done", "completed_texts": total_texts, "progress": 100})

        ################################### BERTopic ###################################

        # Шаг 1 - Очистка данных
        llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

        gc.collect()
        torch.cuda.empty_cache()

        # Шаг 2 - Генерация эмбедингов
        embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
        embeddings = embedding_model.encode(llm_labels, show_progress_bar=True)

        # Шаг 3 - Снижение размерности UMAP
        # umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
        # embeddings_umap = umap_model.fit_transform(embeddings)

        # Проверим, что embeddings не пуст или
        if len(embeddings) > 0:
            umap_model = UMAP(n_neighbors=2, n_components=min(len(embeddings), 5), min_dist=0.0, metric="cosine", random_state=42)
            embeddings_umap = umap_model.fit_transform(embeddings)
        else:
            print("Нет доступных эмбеддингов для обработки.")
        
        # Шаг 4 - Кластеризация HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
        hdbscan_model.fit(embeddings_umap)

        # Вывод результатов кластеризации
        labels = hdbscan_model.labels_

        # Подсчет уникальных значений и их количества
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Число кластеров (кроме шума `-1`)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # Число шумовых точек
        if -1 in unique_labels:
            noise_index = np.where(unique_labels == -1)[0][0]  # Найти индекс метки `-1` в unique_labels
            num_noise_points = counts[noise_index]
        else:
            num_noise_points = 0
        # Шаг 5 - Тематическое моделирование BERTopic
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        topics, probs = topic_model.fit_transform(llm_labels, embeddings)

        # Шаг 6 - Генерация заголовков тем
        topic_labels = topic_model.generate_topic_labels(nr_words=6)  # Например, по 6 ключевых слова на тему
        for i, label in enumerate(topic_labels):
            print(f"Тема {i}: {label}")

        pipe = pipeline(
            model=model,
            tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1,
        )
        # pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

        # Шаг 6 - Генерация заголовков тем
        topic_labels_llama3 = []

        for i, topic in enumerate(topic_model.get_topics().values()):  # Получаем ключевые слова тем
            key_words = " | ".join(token[0] for token in topic[:10])  # Берем 10 ключевых слов для темы

            # Формируем сообщения
            messages = [
                {"role": "system", "content": f"[INST] Используя ключевые слова: {key_words}, сгенерируй на русском языке короткий (до 8 слов) и понятный заголовок для данной темы, пиши только сам заголовок на русском языке. [/INST]"}
            ]
            
            # Очищаем кэш перед вызовом модели
            torch.cuda.empty_cache()
            
            # Используем torch.no_grad() для предотвращения вычисления градиентов
            with torch.no_grad():
                response = pipe(messages, num_return_sequences=1)
            
            # Обрабатываем ответ
            topic_labels_llama3.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', ''))
    
        print(f'topic_labels_llama3_1: {len(topic_labels_llama3)}')

        def shorten_by_words(text, max_words):
            """Сокращает текст до заданного количества слов с добавлением многоточия.

            Аргументы:
            text -- исходный текст (строка)
            max_words -- максимальное количество слов (целое число)

            Возвращает:
            Сокращенный текст (строка).
            """
            words = text.split()  # Разделяем текст на слова
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'  # Сокращаем и добавляем многоточие
            return text  # Если длина не превышает, возвращаем оригинальный текст


        # Сокращение всех тем до 7 слов
        topic_labels_llama3 = [shorten_by_words(topic, 7) for topic in topic_labels_llama3]

        topic_model.set_topic_labels(topic_labels_llama3)
        
        # Визуализация
        fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                        hide_document_hover=False, custom_labels=True)

        # Устанавливаем путь к директории файла
        file_location = f'/home/dev/fastapi/analytics_app/data/{task_data['user_id']}/bertopic_files_directory/{task_data['folder_name']}/'

        # Создание директории, если она не существует
        os.makedirs(os.path.dirname(file_location), exist_ok=True)

        # Формируем новое имя файла с добавлением даты и времени
        new_filename = f"{indexes[int(task_data['index'])]}_{current_time}.html"
        fig.write_html(file_location + new_filename)


        ###################################### save model #################################

        # Название для сохранения файлов
        filename = 'topic_model_' + new_filename.split('.html')[0]

        st = time.time()
        elapsed_time = st - et
        # Получаем целое количество секунд
        total_seconds = int(elapsed_time)

        # Вычисляем часы
        hours = total_seconds // 3600
        # Вычисляем оставшиеся минуты
        minutes = (total_seconds % 3600) // 60
        # Вычисляем оставшиеся секунды
        seconds = total_seconds % 60
        execution_time =  f"{hours} ч. {minutes} мин. {seconds} сек."

        # Теперь сохраняем темы
        try:
            os.chdir(file_location)
            topic_model.save(filename, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

            print(f"Модель успешно сохранена в: {file_location }")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")

        # Сохранение тематик llm_labels
        os.chdir(file_location)
        with open(f'my_list_llm_ans_{indexes[int(task_data['index'])]}_{current_time}.pkl', 'wb') as file:
            pickle.dump(llm_labels, file)

        # Получение и обработка данных пользователя
        user_data = await redis_db.execute_command('HGETALL', task_data['user_id'])
        # Если данные возвращаются в формате 'dict' с байтовыми строками, декодируйте их
        user_data = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}

        # Преобразование строки в объект datetime
        creation_date = datetime.strptime(current_time, "%Y%m%d_%H%M%S")

        # Обработка и сохранение нового результата
        file_info = {
            "html-file": f"{indexes[int(task_data['index'])]}_{current_time}.html",
            "model-file": filename,
            "creation_date": str(creation_date.strftime("%Y-%m-%d %H:%M:%S")),  # Преобразование в формат строки
            "execution_time": execution_time,
            "query_str": task_data['query_str'], 
            "min_date": task_data['min_date'],
            "max_date": task_data['max_date'],
            "index_number": int(task_data['index']),
            "task_id": task_data['task_id'],
            "count_long_texts": count_long_texts
        }

        # Проверяем данные пользователя
        if user_data:
            # Проверка на наличие ключа bertopic_files_directory
            if "bertopic_files_directory" in user_data:
                # Если ключ bertopic_files_directory существует — загружаем его содержимое
                user_folders = json.loads(user_data["bertopic_files_directory"])
            else:
                # Если ключа нет — создаём пустой словарь
                user_folders = {}

            # Проверяем существование папки, переданной в task_data['folder_name']
            folder_name = task_data['folder_name']
            if folder_name in user_folders:
                # Если папка существует, добавляем новый file_info в уже имеющийся список
                user_folders[folder_name].append(file_info)
            else:
                # Если папка не существует, создаём её и добавляем file_info в список
                user_folders[folder_name] = [file_info]

            # Сериализуем обновлённый объект папок (user_folders) в JSON
            serialized_folders = json.dumps(user_folders)

            # Сохраняем обновлённые данные в Redis
            await redis_db.hset(task_data["user_id"], "bertopic_files_directory", serialized_folders)
        else:
            # Если данных пользователя нет, выбрасываем исключение
            raise Exception("User data does not exist.")

    except Exception as e:
        logging.error(f"Ошибка при обработке задачи {task_data['task_id']}: {e}")
        traceback.print_exc()

        # Обновляем статус в случае ошибки
        await redis_db.hset(f"task:{task_data['task_id']}", mapping={"status": "failed", "error": str(e)})

    finally:
        # Сбрасываем статус GPU
        await reset_gpu_status()
        logging.info(f"GPU статус сброшен. Задача {task_data['task_id']} завершена.")