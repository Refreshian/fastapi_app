import os, io, json
import pickle
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


# from bark import SAMPLE_RATE, preload_models

# preload_models(
# text_use_small=True,
# coarse_use_small=True,
# fine_use_gpu=False,
# fine_use_small=True,
# )

################################### data ###################################
et = time.time()

filename = 'Alter_SMI_01.12.2024-03.01.2025.json'
os.chdir('/home/dev/fastapi/analytics_app/data/json_files/adsad/')
with io.open(filename, encoding='utf-8', mode='r') as train_file:
    dict_train = json.load(train_file, strict=False)

texts = [x['text'] for x in dict_train]
print('Всего текстов: {}'.format(len(texts)))

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

model = transformers.AutoModelForCausalLM.from_pretrained(
    "/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto',
)

# Установите pad_token_id
# model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else <выберите_значение>
# model.eval()

################################### promt ###################################

# Исходные промты
system_prompt = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant for labeling topics.
<</SYS>>
"""

example_prompt = """
У меня есть следующий текст:
Отличные новости! В Санкт-Петербурге создается киберклуб для Совета работающей молодежи. Это отличная возможность для всех увлеченных киберспортом найти единомышленников и развиваться вместе! Поддерживаем и ждем новых инициатив! Присоединяйся к «Цифровым героям» - киберспортивному клубу для членов СРМ\nВозможности клуба:\n▫️чат единомышленников;\n▫️сбор команд перед турнирами;\n▫️ участие в соревнованиях (городских, отраслевых, между компаниями и т.д.)\n▫️ обучение по игре Дота 2 с дискордом;\nА ещё впереди кибертурнир по DOTA 2&#33;\n❗️Чтобы присоединиться к клубу - пиши в личку\n[id9173246|Андрею Бородачеву]\n#кибер_СРМ #СРМ_возможности\n#СРМ_увлечения'.

Тема описывается следующими ключевыми словами: 'Санкт-Петербург, киберклуб, молодежь, кибертурнир'.

Основываясь на информации о ключевых словах выше, пожалуйста, выпиши тематики этого текста. Убедитесь, что вы возвращаете только тематики и ничего больше.

[/INST] Открытие киберклуба для молодежи в Санкт-Петербурге
"""


pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1,
)
# Установите pad_token_id
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id 


gc.collect()
torch.cuda.empty_cache()
llm_answer = []

count = 0
# Проходим через каждый текст по отдельности
for i in tqdm(range(len(texts[:50]))):
    single_text = texts[i]

    if len(single_text) < 15000:
    
        # Формируем сообщения
        messages = [
            {"role": "system", "content": system_prompt + example_prompt + 'У меня есть следующий текст: ' + single_text + ' Основываясь на информации о ключевых словах выше, пожалуйста, выпишите тематики этого текста. Убедитесь, что вы возвращаете только тематики и ничего больше. Отвечачй на русском языке.'}
        ]
        
        # Очищаем кэш перед вызовом модели
        torch.cuda.empty_cache()
        
        # Используем torch.no_grad() для предотвращения вычисления градиентов
        with torch.no_grad():
            response = pipe(messages, num_return_sequences=1)
        
        # Обрабатываем ответ
        llm_answer.append(response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', ''))

    else:
        llm_answer.append('Длинный текст')
        count+=1
# os.chdir('/home/dev/fastapi/analytics_app/data/json_files')
# # Сохранение результатов в файл
# with io.open('output_results.json', 'w', encoding='utf-8') as output_file:
#     json.dump(llm_answer, output_file, ensure_ascii=False, indent=4)

# print('Обработка завершена.')

# ################################### BERTopic ###################################

# Pre-calculate embeddings
embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
embeddings = embedding_model.encode(llm_answer, show_progress_bar=True)


import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.model_selection import ParameterGrid

def score_clustering(embeddings, hdbscan_model, umap_model):
    embeddings_reduced = umap_model.fit_transform(embeddings)
    hdbscan_model.fit(embeddings_reduced)
    labels = hdbscan_model.labels_
    
    if len(set(labels)) > 1:
        score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
    else:
        score = -1
    
    return score

# Пример использования
# embeddings = np.random.rand(1000, 10)  # Пример данных
param_grid = {
    'min_cluster_size': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120, 150, 200, 250, 300],
    'n_neighbors': [5, 10, 15, 20, 25, 30]
}

best_score = float('-inf')
best_params = None

for params in ParameterGrid(param_grid):
    hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'])
    umap_model = UMAP(n_neighbors=params['n_neighbors'])
    
    score = score_clustering(embeddings, hdbscan_model, umap_model)
    
    if score > best_score:
        best_score = score
        best_params = params


def find_best_n_components(embeddings, min_components=2, max_components=10):
    best_score = -1
    best_n_components = min_components

    for n_components in range(min_components, max_components + 1):
        # Применяем UMAP с текущим n_components
        umap_model = UMAP(n_components=n_components)
        embeddings_reduced = umap_model.fit_transform(embeddings)
        
        # Обучаем HDBSCAN
        hdbscan_model = HDBSCAN(min_cluster_size=10)
        hdbscan_model.fit(embeddings_reduced)
        
        # Получаем метки кластеров
        labels = hdbscan_model.labels_

        # Убираем шумовые точки, обозначенные -1
        if len(set(labels)) > 1:  # Убедимся, что есть хотя бы один кластер
            score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
            if score > best_score:  # Если новая метрика лучше, сохраняем ее
                best_score = score
                best_n_components = n_components

    return best_n_components, best_score

def score_clustering(embeddings, min_components=2, max_components=10):
    # Находим лучшее значение для n_components
    best_n_components, best_score = find_best_n_components(embeddings, min_components, max_components)
    print(f"Best n_components: {best_n_components}, Score: {best_score}")

    # Теперь применим UMAP с найденным лучшим значением
    umap_model = UMAP(n_components=best_n_components)
    embeddings_reduced = umap_model.fit_transform(embeddings)

    # Создаем и обучаем модель HDBSCAN
    hdbscan_model = HDBSCAN(min_cluster_size=10)
    hdbscan_model.fit(embeddings_reduced)
    
    # Получаем метки кластеров
    labels = hdbscan_model.labels_

    # Убираем шумовые точки, обозначенные -1
    if len(set(labels)) > 1:  # Убедимся, что есть хотя бы один кластер
        score = silhouette_score(embeddings_reduced[labels != -1], labels[labels != -1])
    else:
        score = -1  # В случае отсутствия кластеров или только шумовые точки

    return best_n_components, score

n_components = score_clustering(embeddings=embeddings)
n_components = n_components[0]
print("Лучшие параметры n_components:", n_components)
print("Лучшие параметры:", best_params)

n_neighbors = best_params['n_neighbors']
min_cluster_size = best_params['min_cluster_size']



umap_model = UMAP(n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Pre-reduce embeddings for visualization purposes
reduced_embeddings = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

# Our text generator
generator = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.2,
    max_new_tokens=150,
    # repetition_penalty=1.1
)

# KeyBERT
keybert = KeyBERTInspired()

# MMR
mmr = MaximalMarginalRelevance(diversity=0.3)


# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
[INST]
У меня есть тема, содержащая следующие документы:
[DOCUMENTS]

Тема описывается следующими ключевыми словами: '[KEYWORDS]'.

Основываясь на информации о теме выше, пожалуйста, создайте краткий заголовок этой темы. Убедитесь, что вы возвращаете только заголовок и ничего больше. Отвечай на русском языке.
[/INST]
"""

prompt = system_prompt + example_prompt + main_prompt
# Text generation with Llama 3
llama3_2 = TextGeneration(generator, prompt=prompt)

# All representation models
representation_model = {
    # "KeyBERT": keybert,
    "Llama3": llama3_2,
    # "MMR": mmr,
}


topic_model = BERTopic(

  # Sub-models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(llm_answer, embeddings)

llama3_labels = [label[0][0].split("\n")[0] for label in topic_model.get_topics(full=True)["Llama3"].values()]
topic_model.set_topic_labels(llama3_labels)

# ################################### Visualize ###################################

fig = topic_model.visualize_documents(llm_answer, reduced_embeddings=reduced_embeddings, hide_annotations=True, 
                                hide_document_hover=False, custom_labels=True)

# Модифицируйте метки
# for trace in fig.data:
#     trace.name = ' '.join(trace.name.split()[:10])  # Оставляем только первые 3 слова в метке

os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
fig.write_html("/home/dev/fastapi/analytics_app/data/html_files/" + filename.split('.json')[0] + '.html')

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


    # topic_model.save(filename, save_embeddings=False, save_vectorizer=False)
    # with open(filename + '_model.pkl', 'wb') as f:
    #     pickle.dump(topic_model, f)
    # topic_model.save(topics_file_path)  # используем метод save_topics у модели

    # Создание tar.gz архива
    # archive_name = filename + '.tar.gz'
    # with tarfile.open(archive_name, 'w:gz') as tar:
    #     tar.add(filename, arcname=filename)

    print(f"Модель успешно сохранена в: {save_directory / filename }")
except Exception as e:
    print(f"Ошибка при сохранении модели: {e}")


# os.chdir('/home/dev/fastapi/analytics_app/data/html_files')
# # Сохранение списка в файл с помощью pickle
# with open('my_list.pkl', 'wb') as file:
#     pickle.dump(llm_answer, file)

st = time.time()
elapsed_time = st - et
print('Execution time:', elapsed_time, 'seconds')

print('Длинных текстов:', count)