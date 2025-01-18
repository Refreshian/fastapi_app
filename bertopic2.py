import os, io, json
import pickle
import re
import tarfile
import time
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from torch import cuda

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# model_id = 'meta-llama/Llama-2-7b-chat-hf'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

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
import nltk

# Загружаем списки стоп-слов и токенайзер
nltk.download('stopwords')
nltk.download('punkt')

# Получаем список стоп-слов для русского языка
russian_stopwords = stopwords.words("russian")


# from bark import SAMPLE_RATE, preload_models

# preload_models(
# text_use_small=True,
# coarse_use_small=True,
# fine_use_gpu=False,
# fine_use_small=True,
# )

################################### data ###################################
et = time.time()

os.chdir('/home/dev/fastapi/analytics_app/data/6/json_files_directory/Cyber/')
filename = 'kibersport_01.01.2024-31.12.2024.json'

with io.open(filename, encoding='utf-8', mode='r') as train_file:
    dict_train = json.load(train_file, strict=False)

texts = [x['text'] for x in dict_train]
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
texts = texts[:500]
print('Всего текстов: {}'.format(len(texts)))
# print(texts[:2])

################################### model ###################################

tokenizer = AutoTokenizer.from_pretrained("/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct")

from torch import bfloat16
import transformers

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit quantization
    bnb_4bit_quant_type='nf4',  # Normalized float 4
    bnb_4bit_use_double_quant=True,  # Second quantization after the first
    bnb_4bit_compute_dtype=bfloat16  # Computation type
)



gc.collect()
torch.cuda.empty_cache()

model = transformers.AutoModelForCausalLM.from_pretrained(
    "/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)

################################### promt ###################################

# Исходные промты
system_prompt = """
<s>[INST] <<SYS>>
Ты дружелюбный ассистент-помощник для разметки текстов из соцмедиа на тематики.
<</SYS>>
"""

example_prompt = """
У меня есть следующий текст:
- Отличные новости! В Санкт-Петербурге создается киберклуб для Совета работающей молодежи. Это отличная возможность для всех увлеченных киберспортом найти единомышленников и развиваться вместе! Поддерживаем и ждем новых инициатив! Присоединяйся к «Цифровым героям» - киберспортивному клубу для членов СРМ\nВозможности клуба:\n▫️чат единомышленников;\n▫️сбор команд перед турнирами;\n▫️ участие в соревнованиях (городских, отраслевых, между компаниями и т.д.)\n▫️ обучение по игре Дота 2 с дискордом;\nА ещё впереди кибертурнир по DOTA 2&#33;\n❗️Чтобы присоединиться к клубу - пиши в личку\n[id9173246|Андрею Бородачеву]\n#кибер_СРМ #СРМ_возможности\n#СРМ_увлечения'.

Тема текста описывается следующими ключевыми словами: 'Санкт-Петербург, киберклуб, молодежь'.

Основываясь на информации о теме выше, пожалуйста, создайте краткий заголовок этой темы. Убедитесь, что вы возвращаете только тематики и ничего больше. Отвечайте на русском языке. Пишите только заголовок, не пишите [/INST] в ответе.

[/INST] Открытие киберклуба для молодежи в Санкт-Петербурге
"""


pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.6,
    max_new_tokens=50,
    repetition_penalty=1.2
)
# Установите pad_token_id
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id 


llm_labels = []

count = 0
# Проходим через каждый текст по отдельности
for i in tqdm(range(len(texts))):
    single_text = texts[i]

    if len(single_text) < 45000:
    
        # Формируем сообщения
        messages = [
            {"role": "system", "content": system_prompt + 'У меня есть следующий текст: ' + single_text + ' Основываясь на информации о ключевых словах выше, пожалуйста, выпиши краткий заголовок этого текста. Убедитесь, что вы возвращаете только заголовок и ничего больше. Отвечайте на русском языке. Пишите только заголовок, не пишите [/INST] в ответе.'}
        ]
        
        # Очищаем кэш перед вызовом модели
        torch.cuda.empty_cache()
        
        # Используем torch.no_grad() для предотвращения вычисления градиентов
        with torch.no_grad():
            response = pipe(messages, num_return_sequences=1)
        
        # Обрабатываем ответ
        llm_labels.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', ''))

    else:
        llm_labels.append('Длинный текст')
        count+=1


# ################################### BERTopic ###################################

# Pre-calculate embeddings
# embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
# embeddings = embedding_model.encode(llm_labels, show_progress_bar=True)


# import numpy as np
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# from umap import UMAP
# from hdbscan import HDBSCAN
# from sklearn.model_selection import ParameterGrid

# def score_clustering(embeddings, hdbscan_model, umap_model):
#     embeddings_reduced = umap_model.fit_transform(embeddings)
#     hdbscan_model.fit(embeddings_reduced)
#     labels = hdbscan_model.labels_
    
#     if len(set(labels)) > 1:
#         score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
#     else:
#         score = -1
    
#     return score

# # Пример использования
# # embeddings = np.random.rand(1000, 10)  # Пример данных
# param_grid = {
#     'min_cluster_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300],
#     'n_neighbors': [3, 5, 7, 8, 9, 10, 12, 15, 18, 20, 23, 25, 30, 35, 40, 50, 60, 70, 80]
# }

# best_score = float('-inf')
# best_params = None

# for params in ParameterGrid(param_grid):
#     hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'])
#     umap_model = UMAP(n_neighbors=params['n_neighbors'])
    
#     score = score_clustering(embeddings, hdbscan_model, umap_model)
    
#     if score > best_score:
#         best_score = score
#         best_params = params


# def find_best_n_components(embeddings, min_components=2, max_components=10):
#     best_score = -1
#     best_n_components = min_components

#     for n_components in range(min_components, max_components + 1):
#         # Применяем UMAP с текущим n_components
#         umap_model = UMAP(n_components=n_components)
#         embeddings_reduced = umap_model.fit_transform(embeddings)
        
#         # Обучаем HDBSCAN
#         hdbscan_model = HDBSCAN(min_cluster_size=10)
#         hdbscan_model.fit(embeddings_reduced)
        
#         # Получаем метки кластеров
#         labels = hdbscan_model.labels_

#         # Убираем шумовые точки, обозначенные -1
#         if len(set(labels)) > 1:  # Убедимся, что есть хотя бы один кластер
#             score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
#             if score > best_score:  # Если новая метрика лучше, сохраняем ее
#                 best_score = score
#                 best_n_components = n_components

#     return best_n_components, best_score

# def score_clustering(embeddings, min_components=2, max_components=10):
#     # Находим лучшее значение для n_components
#     best_n_components, best_score = find_best_n_components(embeddings, min_components, max_components)
#     print(f"Best n_components: {best_n_components}, Score: {best_score}")

#     # Теперь применим UMAP с найденным лучшим значением
#     umap_model = UMAP(n_components=best_n_components)
#     embeddings_reduced = umap_model.fit_transform(embeddings)

#     # Создаем и обучаем модель HDBSCAN
#     hdbscan_model = HDBSCAN(min_cluster_size=10)
#     hdbscan_model.fit(embeddings_reduced)
    
#     # Получаем метки кластеров
#     labels = hdbscan_model.labels_

#     # Убираем шумовые точки, обозначенные -1
#     if len(set(labels)) > 1:  # Убедимся, что есть хотя бы один кластер
#         score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
#     else:
#         score = -1  # В случае отсутствия кластеров или только шумовые точки

#     return best_n_components, score

# n_components = score_clustering(embeddings=embeddings)
# n_components = n_components[0]
# print("Лучшие параметры n_components:", n_components)
# print("Лучшие параметры:", best_params)

# n_neighbors = best_params['n_neighbors']
# min_cluster_size = best_params['min_cluster_size'] 



# umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
# hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# # Pre-reduce embeddings for visualization purposes
# reduced_embeddings = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

# # Our text generator
# generator = transformers.pipeline(
#     model=model, tokenizer=tokenizer,
#     task='text-generation',
#     temperature=0.6,
#     max_new_tokens=50,
#     # repetition_penalty=1.2
# )

# # KeyBERT
# keybert = KeyBERTInspired()

# # MMR
# mmr = MaximalMarginalRelevance(diversity=0.3)


# # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
# main_prompt = """
# [INST]
# У меня есть тема, содержащая следующие документы:
# [DOCUMENTS]

# Тема описывается следующими ключевыми словами: '[KEYWORDS]'.

# Основываясь на информации о теме выше, пожалуйста, создайте краткий заголовок этой темы. Убедитесь, что вы возвращаете только заголовок и ничего больше. Отвечайте на русском языке. Не пишите [/INST] в ответе.
# [/INST]
# """

# prompt = system_prompt + example_prompt + main_prompt
# # Text generation with Llama 3
# llama3_2 = TextGeneration(generator, prompt=prompt)

# # All representation models
# representation_model = {
#     # "KeyBERT": keybert,
#     "Llama3": llama3_2,
#     # "MMR": mmr,
# }


# topic_model = BERTopic(

#   # Sub-models
#   embedding_model=embedding_model,
#   umap_model=umap_model,
#   hdbscan_model=hdbscan_model,
#   representation_model=representation_model,

#   # Hyperparameters
#   top_n_words=10,
#   verbose=True
# )

# # Train model
# llm_labels = [x.replace('[/INST]', '').replace('[INST]', '').replace('INST', '').replace(']', '').replace('[', '')
#                for x in llm_labels]

# print('777888999')
# print(llm_labels)

# topics, probs = topic_model.fit_transform(llm_labels, embeddings)

# llama3_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama3"].values()]
# topic_model.set_topic_labels(llama3_labels)
# print('555+++555')
# topic_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama3"].values()]
# print(topic_labels)

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import re
from bertopic import BERTopic

# Шаг 1 - Очистка данных
llm_labels = [re.sub(r"[^\w\s\"«»']", "", label.strip()) for label in llm_labels if label.strip()]

# Шаг 2 - Генерация эмбедингов
embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
embeddings = embedding_model.encode(llm_labels, show_progress_bar=True)

# Шаг 3 - Снижение размерности UMAP
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
embeddings_umap = umap_model.fit_transform(embeddings)

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
topic_labels = topic_model.generate_topic_labels(nr_words=5)  # Например, по 3 ключевых слова на тему
for i, label in enumerate(topic_labels):
    print(f"Тема {i}: {label}")

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.6,
    max_new_tokens=50,
    repetition_penalty=1.1,
)
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

# Шаг 6 - Генерация заголовков тем
topic_labels_llama3 = []

for i, topic in enumerate(topic_model.get_topics().values()):  # Получаем ключевые слова тем
    key_words = " | ".join(token[0] for token in topic[:10])  # Берем 5 ключевых слов для темы

    # Формируем сообщения
    messages = [
        {"role": "system", "content": f"[INST] Используя данные ключевые слова: {key_words}, сгенерируй на русском языке короткий (не более 5-7 слов) и понятный заголовок для данной темы. Не пиши какие ключевые слова ты использовал (Using keywords), не пиши дополнительных пояснений для заголовка, пиши только сам заголовок на русском языке. [/INST]"}
    ]
    
    # Очищаем кэш перед вызовом модели
    torch.cuda.empty_cache()
    
    # Используем torch.no_grad() для предотвращения вычисления градиентов
    with torch.no_grad():
        response = pipe(messages, num_return_sequences=1)
    
    # Обрабатываем ответ
    topic_labels_llama3.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', ''))

    # generated_label = response[0]['generated_text'].replace('[/INST]\n', '').replace('\n', '').replace('[/INST]', '')
    # topic_labels_llama3.append(f"Тема {i}: {generated_label}")

for i, label in enumerate(topic_labels_llama3):
    print(f"Тема {i}: {label}")

topic_model.set_topic_labels(topic_labels_llama3)

# Шаг 7 - Визуализации
topic_model.visualize_topics()
# topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap)

# ################################### Visualize ###################################

fig = topic_model.visualize_documents(llm_labels, reduced_embeddings=embeddings_umap, hide_annotations=True, 
                                hide_document_hover=False, custom_labels=True)

# Модифицируйте метки
# for trace in fig.data:
#     trace.name = ' '.join(trace.name.split()[:10])  # Оставляем только первые 3 слова в метке

print(555777999)
print(filename)
os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
fig.write_html(filename.split('.json')[0] + '.html')

###################################### save model #################################

from pathlib import Path
from PIL import Image
import joblib  # или import pickle
# Задайте директорию для сохранения файлов
filename = 'topic_model_' + filename.split('.json')[0]
save_directory = Path("/home/dev/fastapi/analytics_app/data/html_files")  # укажите путь к директории
topics_file_path = save_directory / filename
print(topics_file_path)

# Проверяем, существует ли файл и удаляем его
if topics_file_path.exists():
    os.remove(topics_file_path)
    print(f"Удален старый файл: {topics_file_path}")

# Теперь сохраняем темы
try:
    # joblib.dump(model, 'bertopic_model.joblib')
    os.chdir(save_directory)
    topic_model.save(filename, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)


    print(f"Модель успешно сохранена в: {save_directory / filename }")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")


os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
# Сохранение списка в файл с помощью pickle
with open('my_list_llm_ans.pkl', 'wb') as file:
    pickle.dump(llm_labels, file)

st = time.time()
elapsed_time = st - et
print('Execution time:', elapsed_time, 'seconds')

print('Длинных текстов:', count)