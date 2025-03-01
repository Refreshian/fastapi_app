# import os
# import logging
# import os
# import re
# import time
# import pickle
# import asyncio
# import gc
# import json
# import traceback
# import aiohttp
# import torch
# from tqdm import tqdm
# from datetime import datetime
# from transformers import AutoTokenizer, pipeline
# from sentence_transformers import SentenceTransformer
# from concurrent.futures import ThreadPoolExecutor
# from sklearn.metrics import silhouette_score
# from umap import UMAP
# from hdbscan import HDBSCAN
# from bertopic import BERTopic
# import pandas as pd
# import datamapplot

# from torch import bfloat16
# from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
# from search_data_elastic import elastic_query
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy.ext.asyncio import create_async_engine


# data = elastic_query(
#     theme_index='tehno_shuffle_25',
#     query_str='all',
#     min_date=1706734800,
#     max_date=1737406722
# )

# df = pd.DataFrame(data)

# len(data)


# file_location = f'/home/dev/fastapi/analytics_app/files/'
# os.chdir(file_location)
# df.to_excel('df_meta_testing_25_fobii.xlsx', index=False)


import os
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

os.chdir('/home/dev/fastapi/analytics_app/files/fobii_join_embed.xlsx')
df_all = pd.read_excel('calls_zip.xlsx')
df = df_all.dropna(subset=['Тема'])
print(df.shape)
print(df.head())

embedding_model = SentenceTransformer("/home/dev/fastapi/analytics_app/data/embed_files/DeepPavlov/rubert-base-cased-sentence") # 768-hidden

# Функция для обработки эмбеддингов части данных
def process_embeddings(labels_chunk):
    embeddings_chunk = []
    for label in labels_chunk:
        embedding = embedding_model.encode(label, show_progress_bar=False)
        embeddings_chunk.append(embedding)
    
    # Возвращаем эмбеддинги как NumPy массив для дальнейшей обработки
    return np.array(embeddings_chunk)

# Разделяем данные на пакеты
df_dict = df.groupby('Тема')['operator_text'].apply(list).to_dict()
df_dict = {key: value for key, value in df_dict.items() if len(value) > 100}

# Максимальная длина токенов
max_length = 512

# Функция для получения эмбеддингов
def get_embeddings(text):
    # Если текст длиннее максимальной длины, разбиваем его на кусочки
    text_pieces = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # Извлечение эмбеддингов для всех частей текста
    if text_pieces:  # Проверяем, что есть кусочки текста
        embeddings = embedding_model.encode(text_pieces, show_progress_bar=False)
        # Возвращаем усредненный эмбеддинг
        return np.mean(embeddings, axis=0)
    else:
        # Если нет частей, возвращаем нулевой вектор
        return np.zeros((768,))  # Размер эмбеддинга, предположительно 768

# Словарь для хранения эмбеддингов
embeddings_key = {}

# Извлечение эмбеддингов для ключей и значений
count = 0
for key, texts in tqdm(df_dict.items()):
    # count += 1
    # if count > 1:  # Здесь вы можете удалить или изменить условие для обработки всех тем
    #     break
    
    # Эмбеддинги ключа
    key_embedding = embedding_model.encode(key).flatten()
    
    # Эмбеддинги значений
    text_embeddings = [get_embeddings(text) for text in texts]

    # Сохраняем в словаре: эмбеддинг ключа и массив эмбеддингов для значений
    embeddings_key[key] = {
        'key_embedding': key_embedding,
        'text_embeddings': text_embeddings
    }

# Сохранение словаря в файл
with open("embeddings_key_fobii.pkl", "wb") as file:
    pickle.dump(embeddings_key, file)

print("Словарь успешно сохранен в файл embeddings_key_fobii.pkl")

# # Загрузка словаря из файла
# with open("embeddings_key.pkl", "rb") as file:
#     loaded_embeddings_key = pickle.load(file)

# print("Словарь успешно загружен из файла:", loaded_embeddings_key)