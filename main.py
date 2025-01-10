import ast
import asyncio
from datetime import datetime
from enum import Enum
import gc
import glob
import itertools
import re
import shutil
from typing import List, Optional, Union, Dict
from collections import ChainMap, defaultdict
import time
from os import listdir
from os.path import isfile, join
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

import aiofiles
from sklearn import manifold
from fastapi_users import fastapi_users, FastAPIUsers
import pandas as pd
from pydantic import BaseModel, Field
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import BackgroundTasks, FastAPI, File, Request, UploadFile, WebSocket, logger, status, Depends
from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np

import functools as ft
import io

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import codecs, json

import websocket

from auth.auth import auth_backend
from auth.database import User
from auth.manager import get_user_manager
from auth.schemas import UserRead, UserCreate
from fastapi.middleware.cors import CORSMiddleware 
from elasticsearch import Elasticsearch, helpers
import sys, json, os
from load_data_elastic import load_file_to_elstic
from search_data_elastic import elastic_query
from operator import itemgetter
from transformers import AutoTokenizer, pipeline
import torch

import tensorflow_hub as hub
import tensorflow_text
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

import jwt
from sqlalchemy.orm import Session 
from fastapi import HTTPException, status
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from fastapi_users.db import SQLAlchemyBaseUserTable
from sqlalchemy import Column, String, Boolean, Integer, TIMESTAMP, ForeignKey

from datetime import datetime
from typing import AsyncGenerator
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from config import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
from model.models import role
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tarfile
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from torch import cuda

from torch import bfloat16
import transformers

from umap import UMAP
from hdbscan import HDBSCAN
import gc
import torch, os, json
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration

from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from pathlib import Path
from PIL import Image
import joblib  # import pickle
import tensorflow as tf


# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:50"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_async_engine(DATABASE_URL)
async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Секретный ключ
SECRET_KEY = "SECRET"
ALGORITHM = "HS256"  # Указание алгоритма

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


import logging
# Настройка логирования для записи в файл
logging.basicConfig(filename='app.log', level=logging.INFO)

import redis
# redis_db = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True) # БД  для прогресс-бара с LLM расчетами
# Инициализация клиента Redis
redis_db = redis.Redis(host='localhost', port=6379, db=0)


es = Elasticsearch(
    ['localhost'],
    port=9200
)


path_json_files = '/home/dev/fastapi/fastapi_app/data/json_files'

app = FastAPI(
    title="Analytics App"
)


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Настройка CORS
origins = [ 
    "http://localhost",
    "http://localhost:5000",
    "http://localhost:5173",
    "http://194.146.113.123:5000",  # Добавьте ваш IP адрес
    "http://localhost:5174",
    "http://194.146.113.123",
    "https://194.146.113.123"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Или укажите конкретные методы
    allow_headers=["*"],  # Или укажите конкретные заголовки
)

# from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
# app.add_middleware(HTTPSRedirectMiddleware)

# db

torch.cuda.empty_cache() 
gc.collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

# load LLM
# os.chdir('/home/dev/fastapi/fastapi_app/data/LLM_models')

# model = "gemma-2b-it"
# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = pipeline(
#     "text-generation",
#     model=model,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# ) 
 

fastapi_users = FastAPIUsers[User, int]( 
    get_user_manager,
    [auth_backend], 
)
 
### TonalityLandscape Models
class TonalityValues(BaseModel):
    negative_count: int 
    positive_count: int

class NegativeHub(BaseModel):
    name: str
    values: int

class PositiveHub(BaseModel):
    name: str
    values: int


class ModelAuthorsTonalityLandscape(BaseModel):
    negative_hubs: List[NegativeHub]
    positive_hubs: List[PositiveHub]
    

class Text(BaseModel):
    text: str
    hub: str
    url: str
    er: int
    viewsCount: Union[int, str]
    region: str


class Text(BaseModel):
    text: str
    hub: str
    url: str
    er: int
    viewsCount: Union[int, str]
    region: str


class AuthorDatum(BaseModel):
    fullname: str
    url: str
    author_type: str
    sex: str
    age: str
    count_texts: int
    texts: List[List[Text]]


class ModeAuthorValues(BaseModel):
    author_data: List[AuthorDatum]


class Model_TonalityLandscape(BaseModel):
    tonality_values: TonalityValues
    tonality_hubs_values: ModelAuthorsTonalityLandscape
    negative_authors_values: List[ModeAuthorValues]
    positive_authors_values: List[ModeAuthorValues]
###=====###

### Information Graph Models
class AuthorInfGraph(BaseModel):
    fullname: str
    url: str
    author_type: str
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: Union[int, str]
    timeCreate: str


class RepostInfGraph(BaseModel):
    fullname: str
    url: str
    author_type: str
    sex: str
    age: str
    audienceCount: int
    er: int
    viewsCount: str
    timeCreate: str


class AuthorsStream(BaseModel):
    author: AuthorInfGraph
    reposts: List[RepostInfGraph]


class ModelInfGraph(BaseModel):
    values: List[AuthorsStream]
    dynamicdata_audience: dict
    post: bool
    repost: bool
    SMI: bool


# Themes Model
class ThemesValues(BaseModel):
    description: str
    count: int
    audience: str
    er: str
    viewsCount: str
    texts: str


class ThemesModel(BaseModel):
    values: List[ThemesValues]

# Customer Voice Model
class TonalityVoice(BaseModel):
    source: str
    Нейтрал: int
    Позитив: int
    Негатив: int


class SunkeyDatum(BaseModel):
    hub: str
    type: str
    tonality: str
    count: int
    search: str


class VoiceModel(BaseModel):
    name: str
    tonality: List[TonalityVoice]
    sunkey_data: List[SunkeyDatum]


class ModelVoice(BaseModel):
    __root__: List[VoiceModel]

# Mediarating Model
class NegativeSmiMediaRating(BaseModel):
    name: str
    index: int
    message_count: int


class PositiveSmiMediaRating(BaseModel):
    name: str
    index: int
    message_count: int


class FirstGraphMediaRating(BaseModel):
    negative_smi: List[NegativeSmiMediaRating]
    positive_smi: List[PositiveSmiMediaRating]


class SecondGraphItemMediaRating(BaseModel):
    name: str
    time: int
    index: int
    url: str
    color: str


class MediaRatingModel(BaseModel):
    first_graph: FirstGraphMediaRating
    second_graph: List[SecondGraphItemMediaRating]


class ModelItemAIAnalyticsNone(BaseModel):
    id: int
    timeCreate: int
    text: str
    hub: str
    audienceCount: int
    commentsCount: int
    er: int
    url: str

# ModelAiAnalytics
class ModelAiAnalyticsItem(BaseModel):
    id: int
    timeCreate: int
    text: str
    hub: str
    audienceCount: None
    commentsCount: None
    er: None
    url: str


class ModelAiAnalytics(BaseModel):
    data: List[ModelAiAnalyticsItem]


# class ModelAIPostAnalytics(BaseModel):
#     id: int
#     text: str
#     llm_text: str


# class ModelAIAnalyticsPost(BaseModel):
#     promt: str
#     texts: List[ModelAIPostAnalytics]


class QueryAiLLM(BaseModel):
    index: int=None
    min_date: int=None
    max_date: int=None
    promt: str = None
    texts_ids: list[int] = None


### Model Competitors
class QueryCompetitors(BaseModel):
    themes_ind: List[int] = Field(default_factory=list)
    min_date: Optional[int] = None
    max_date: Optional[int] = None


class FirstGraphItem(BaseModel):
    index_name: str
    values: List


class NegItem(BaseModel):
    hub: str
    audienceCount: int


class Po(BaseModel):
    hub: str
    audienceCount: int


class SMI(BaseModel):
    name: str
    neg: List[NegItem]
    pos: List[Po]


class SecondGraphItem(BaseModel):
    index_name: str
    SMI: SMI


class SMIItem(BaseModel):
    name: str
    count: int
    rating: Optional[int]


class SocmediaItem(BaseModel):
    name: str
    count: int
    rating: Optional[int]


class ThirdGraphItem(BaseModel):
    index_name: str
    SMI: List[SMIItem]
    Socmedia: List[SocmediaItem]


class CompetitorsModel(BaseModel):
    first_graph: List[FirstGraphItem]
    second_graph: List[SecondGraphItem]
    third_graph: List[ThirdGraphItem]


class DataFolder(BaseModel):
    name: str
    values: List[str]


class ModelDataFolder(BaseModel):
    values: List[DataFolder]

###=====###

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"], 
)
 
current_user = fastapi_users.current_user()

# indexes = {1: "rosbank_01.02.2024-07.02.2024", 2: "skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024", 3:'rosbank_19.02.2024-29.02.2024', 
#            4: "rosbank_14.03.2024-14.03.2024_fullday", 5: "r_13.03.2024-14.03.2024_full", 6: "rosbank_22.03.2024-24.03.2024", 
#            7: "monitoring_tem_19.03.2024-25.03.2024", 8: 'rosbank_26.03.2024-01.04.2024', 9: 'tehfob', 10: 'transport_01.01.2024-09.04.2024', 
#            11: 'moskovskiy_transport_01.01.2024_09.04.2024_2b', 12: 'rosbank_01.04.2024-15.04.2024', 13: 'rosbank_14.05.2024-16.05_чистая прибыль',
#            14: 'contented_smi_01.04.2024-26.05.2024', 15: 'skillbox_smi_01.04.2024-26.05.2024', 16: 'rb_smi', 17: 'geekbrains', 18: 'eduson', 
#            19: 'maley_nlmk_boevaya_tema_17.06.2024-21.06.2024_66757eb24cb15033866ecdd8', 20: 'maley_nlmk_boevaya_tema_17_06_2024_21_06_2024',
#            21: 'platon_test_31.07.2024-06.08.2024', 22: 'platon_test', 23: 'avtomobili_01.09.2023-02.09.2024', 24: 'cennosti_01.08.2024-31.08.2024', 
#            25: 'cennosti_01.07.2024-31.07.2024', 26: 'cennosti_data_year', 27: 'cennosti_data_year_without_doubles', 28: 'irkutsk', 
#            29: 'platon_22.11.2024-21.12.2024'}

# сохранение начального словаря всех файлов/тем
def save_dict_to_pickle(file_path, data_dict):
    """
    Сохраняет словарь в файл с использованием Pickle.
    :param file_path: Путь к файлу, куда нужно сохранить словарь (str).
    :param data_dict: Словарь, который нужно сохранить (dict).
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Словарь успешно сохранен в {file_path}.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении файла: {e}")

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

 
# @app.get('/data-users')
# async def data_users(): # user: User = Depends(current_user)

#     es_indexes = [index for index in es.indices.get('*')] # список всех индексов elastic
#     es_indexes = [x.strip() for x in es_indexes]

#     # поиск мин и макс дат в данных/файлах
#     query = {

#     "aggs": { 
#         "max_timeCreate": {
#         "max": {
#             "field": "timeCreate"
#         }
#         },
#         "min_timeCreate": {
#         "min": {
#             "field": "timeCreate"
#         }
#         }
#     },
#     }

#     # Путь к файлу с темами 
#     file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
#     # Загрузка словаря с темами
#     indexes = load_dict_from_pickle(file_path)

#     data_index = []
#     for index in es_indexes:
#         if index == 'read_me':
#             continue
#         date_period_query = es.search(index=index, body=query)['aggregations'] # запрос мин и макс дат в индексе
#         try:
#             data_index.append(
#                 { 
#                     "file": index,
#                     "min_data": date_period_query['min_timeCreate']['value'],
#                     "max_data": date_period_query['max_timeCreate']['value'],
#                     "index_number": list({i for i in indexes if indexes[i]==index})[0]
#                 }
#             )
#         except:
#             continue
    
#     data_index = sorted(data_index, key=lambda d: d['index_number'])
#     return {"values": data_index}


@app.get("/tonality_landscape")
async def tonality_landscape(user: User = Depends(current_user), index: int =None, 
                             min_date: int=None, max_date: int=None) -> Model_TonalityLandscape:

    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # запрос к данным для по запрашиваемому индексу/теме
    # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-range-query.html
    query = {
            "size": 10000,
            "query": {
                        "range": {
                            "timeCreate": {      # skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024
                                "gte": min_date, # 1705329992
                                "lte": max_date, # 1705848392
                                "boost": 2.0
                            }
                        }
                    }
                }
    # print('+++===+++')
    # print(index)
    # min_date = 1705329992
    # max_date = 1705848392
    # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', body=query)
    data = es.search(index=indexes[index], body=query)
    data = data['hits']['hits']

    ### подсчет количества позитива и негатива
    pos = [x['_source']['toneMark'] for x in data if x['_source']['toneMark'] == 1]
    neg = [x['_source']['toneMark'] for x in data if x['_source']['toneMark'] == -1]

    ### подсчет источников
    # негатив
    neg_hub = [x['_source']['hub'] for x in data if x['_source']['toneMark'] == -1]
    dct_neg_hub = dict(Counter(neg_hub)) 
    dct_neg_hub = dict(sorted(dct_neg_hub.items(), key=lambda x:x[1], reverse=True)) # {'telegram.org': 4, 'vk.com': 3, 'ok.ru': 1}
    # позитив
    pos_hub = [x['_source']['hub'] for x in data if x['_source']['toneMark'] == 1]
    dct_pos_hub = dict(Counter(pos_hub)) 
    dct_pos_hub = dict(sorted(dct_pos_hub.items(), key=lambda x:x[1], reverse=True))
    # dct_pos_hub = json.dumps(dct_pos_hub)

    ### получение данных для ландшафта авторов по позитиву и негативу

    ## авторы негатива
    neg_authors = [x['_source'] for x in data if x['_source']['toneMark'] == -1]
    pos_authors = [x['_source'] for x in data if x['_source']['toneMark'] == 1]

    # группировка авторов по истонику (hub)
    neg_authors_hub = []
    for key in dct_neg_hub.keys():
        neg_authors_hub.append([(x['authorObject'], [{"text": x['text'], "hub": x['hub'], "url": x['url'], "er": x['er'], 
                                "viewsCount": x['viewsCount'], "region": x['region']}]) for x in neg_authors if x['hub'] == key])
    
    # получение итогового словаря по негативным авторам с учетом данных сколько текстов написал автор
    a = []
    for i in range(len(neg_authors_hub)):
        name_unique_author = [x[0]['fullname'] if 'fullname' in x[0] else neg_authors_hub[i][0][1][0]['hub'] for x in neg_authors_hub[i]]
        dct_non_unique_author = dict(Counter(name_unique_author))
        list_non_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val > 1]))
        list_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val == 1]))

        # если есть неуникальные авторы (несколько текстов от автора за период)
        if list_non_unique_authors != []:
            for k in range(len(list_non_unique_authors)):
                c ={}
                c['author_data'] = []
                # забираем словарь с authorobject
                try:
                    author_dict = [x[0] for x in neg_authors_hub[i] if x[0]['fullname'] == list_non_unique_authors[k]][0]
                    texts = [x[1] for x in neg_authors_hub[i] if x[0]['fullname'] == list_non_unique_authors[k]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)

            #         добавляем тексты автора
                    author_dict['texts'] = texts

                except:
                    author_dict = {'fullname': neg_authors_hub[i][0][1][0]['hub'], 'url': neg_authors_hub[i][0][1][0]['hub'], 
                                'author_type': 'СМИ', 'sex': '', 'age': ''}
                    texts = [x[1] for x in neg_authors_hub[i] if x[1][0]['hub'] == list_non_unique_authors[k]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)
            #         добавляем тексты автора
                    author_dict['texts'] = texts

                c['author_data'].append(author_dict)
                a.append(c)

        if list_unique_authors != []:
            # сбор уникальных (с одним текстом за период) авторов
            
            for u in range(len(list_unique_authors)):
                c ={}
                c['author_data'] = []
                # забираем словарь с authorobject
                try:
                    author_dict = [x[0] for x in neg_authors_hub[i] if x[0]['fullname'] == list_unique_authors[u]][0]
                    texts = [x[1] for x in neg_authors_hub[i] if x[0]['fullname'] == list_unique_authors[u]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)

            #         добавляем тексты автора
                    author_dict['texts'] = texts

                except:
                    author_dict = {'fullname': neg_authors_hub[i][0][1][0]['hub'], 'url': neg_authors_hub[i][0][1][0]['hub'], 
                                'author_type': 'СМИ', 'sex': '', 'age': ''}
                    texts = [x[1] for x in neg_authors_hub[i] if x[1][0]['hub'] == list_unique_authors[u]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)
            #         добавляем тексты автора
                    author_dict['texts'] = texts

                c['author_data'].append(author_dict)
                a.append(c)

    ## авторы позитива
    pos_authors = [x['_source'] for x in data if x['_source']['toneMark'] == 1]

    # группировка авторов по истонику (hub)
    pos_authors_hub = []
    for key in dct_pos_hub.keys():
        pos_authors_hub.append([(x['authorObject'], [{"text": x['text'], "hub": x['hub'], "url": x['url'], "er": x['er'], 
                                "viewsCount": x['viewsCount'], "region": x['region']}]) for x in pos_authors if x['hub'] == key])
    
    # получение итогового словаря по позитивным авторам с учетом данных сколько текстов написал автор
    ### получение данных для ландшафта авторов по позитиву и негативу

    ## авторы позитива
    # группировка авторов по истонику (hub)
    pos_authors_hub = [] 
    for key in dct_pos_hub.keys():
        pos_authors_hub.append([(x['authorObject'], [{"text": x['text'], "hub": x['hub'], "url": x['url'], "er": x['er'], 
                                    "viewsCount": x['viewsCount'], "region": x['region']}]) for x in pos_authors if x['hub'] == key])

    # получение итогового словаря по негативным авторам с учетом данных сколько текстов написал автор
    d = []
    for i in range(len(pos_authors_hub)):
        
        name_unique_author = [x[0]['fullname'] if 'fullname' in x[0] else pos_authors_hub[i][0][1][0]['hub'] for x in pos_authors_hub[i]]
        dct_non_unique_author = dict(Counter(name_unique_author))
        list_non_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val > 1]))
        list_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val == 1]))

        # если есть неуникальные авторы (несколько текстов от автора за период)
        if list_non_unique_authors != []:
            list_non_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val > 1]))
            for k in range(len(list_non_unique_authors)):
                
                c ={}
                c['author_data'] = []
                # забираем словарь с authorobject
                try:
                    author_dict = [x[0] for x in pos_authors_hub[i] if x[0]['fullname'] == list_non_unique_authors[k]][0]
                    texts = [x[1] for x in pos_authors_hub[i] if x[0]['fullname'] == list_non_unique_authors[k]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)

            #         добавляем тексты автора
                    author_dict['texts'] = texts

                except:
                    author_dict = {'fullname': pos_authors_hub[i][0][1][0]['hub'], 'url': pos_authors_hub[i][0][1][0]['hub'], 
                                'author_type': 'СМИ', 'sex': '', 'age': ''}
                    texts = [x[1] for x in pos_authors_hub[i] if x[1][0]['hub'] == list_non_unique_authors[k]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)
            #         добавляем тексты автора
                    author_dict['texts'] = texts

                c['author_data'].append(author_dict)
                d.append(c)

        if list_unique_authors != []:
            # сбор уникальных (с одним текстом за период) авторов
            list_unique_authors = list(set([key for key, val in dct_non_unique_author.items() if val == 1]))
            for u in range(len(list_unique_authors)):
                c ={}
                c['author_data'] = []
                # забираем словарь с authorobject
                try:
                    author_dict = [x[0] for x in pos_authors_hub[i] if x[0]['fullname'] == list_unique_authors[u]][0]
                    texts = [x[1] for x in pos_authors_hub[i] if x[0]['fullname'] == list_unique_authors[u]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)

            #         добавляем тексты автора
                    author_dict['texts'] = texts

                except:
                    author_dict = {'fullname': pos_authors_hub[i][0][1][0]['hub'], 'url': pos_authors_hub[i][0][1][0]['hub'], 
                                'author_type': 'СМИ', 'sex': '', 'age': ''}
                    texts = [x[1] for x in pos_authors_hub[i] if x[1][0]['hub'] == list_unique_authors[u]] # тексты автора за период
                    author_dict['count_texts'] = len(texts)
            #         добавляем тексты автора
                    author_dict['texts'] = texts

                c['author_data'].append(author_dict)
                d.append(c)

    lst_items = list(dct_pos_hub.items())
    dct_pos_hub = [{"name": x[0], "values": x[1]} for x in lst_items]

    lst_items = list(dct_neg_hub.items())
    dct_neg_hub = [{"name": x[0], "values": x[1]} for x in lst_items]

    values = {}
    values['negative_count'] = len(neg)
    values['positive_count'] = len(pos)

    values['dct_neg_hub'] = dct_neg_hub
    values['dct_pos_hub'] = dct_pos_hub

    values['neg_authors'] = a
    values['pos_authors'] = d

    # return values
    values = Model_TonalityLandscape(tonality_values={"negative_count": len(neg), "positive_count": len(pos)}, 
                 tonality_hubs_values={"negative_hubs": dct_neg_hub, "positive_hubs": dct_pos_hub}, negative_authors_values=a, positive_authors_values=d)
    return values


@app.get('/information_graph')
async def information_graph(index: int=None, 
                             min_date: int=None, max_date: int=None, query_str: Optional[str] = 'карта', 
                             post: Optional[bool] = None, repost: Optional[bool] = None, 
                             SMI: Optional[bool] = None) -> ModelInfGraph:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str=query_str)
    # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', query_str='data')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    
    if post == None:
        post = False
    if repost == None:
        repost = False
    if SMI == None:
        SMI = False

    # предобработка данных
    df_meta = pd.DataFrame(data)
    # del data

    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(
        df_meta['text'].values)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)

    dff = pd.DataFrame(cosine_similarity_matrix)

    val_dff = dff.values
    # заменяем значения по главной диагонали на 0
    for i in range(len(val_dff)):
        val_dff[i][i] = 0
        
    dff = pd.DataFrame(val_dff)

    df_meta = df_meta.join(pd.DataFrame(list(df_meta['authorObject'].values), columns=['fullname', 'text_url', 'author_type', 'sex', 'age']))
    # заменяем пустые fullname в СМИ на значения из hub
    df_meta['fullname'].fillna(df_meta['hub'], inplace=True)
    df = df_meta.copy()

    # создаем словарь похожих текстов вида {12: [11, 13],  44: [190], ...}
    fin_dict = {}
    threashhold = 0.8

    # выявляем список строк с похожими текстам
    for i in range(dff.shape[0]):
        if list(np.where(dff.loc[i].values >= threashhold)[0]) != []:
            if i not in [item for sublist in list(fin_dict.values()) for item in sublist]:
                #             if list(np.where(dff.loc[i].values >= threashhold)[0]) in fin_dict.values():
                #                 fin_dict[list(fin_dict.keys())[list(fin_dict.values()).index(list(np.where(dff.loc[i].values >= threashhold)[0]))]].append(i)
                #             else:
                fin_dict[i] = list(
                    np.where(dff.loc[i].values >= threashhold)[0])
                
        else:
            fin_dict[i] = []
            
            
    df_meta.fillna('', inplace=True)
    # оставляем необходимую мету
    df_meta = df_meta[['fullname', 'url', 'author_type', 'hub', 'sex', 'age', 'audienceCount', 'er', 'viewsCount', 'timeCreate']]


    # получение итогового массива данных с последовательностями авторов распространения информации и репостами (похожими текстами)
    data = []

    for key, val in fin_dict.items():
        author_dct = {}
        # забираем отдельно автора и метаданные по нему
        author_dct['author'] = df_meta.loc[key].to_dict()
        # присоединяем репосты к автору, если похожие тексты были далее 
        author_dct['reposts'] = []
        
        if len(val) > 0:
            for i in range(len(val)):
                author_dct['reposts'].append(df_meta[df_meta.index.isin([val[i]])].T.to_dict()[val[i]]) # добавляем словарь с автором репоста и его метаданными
        else:
            pass
        
        data.append(author_dct)

    ### данные для динамического графика
    def to_datetime(unixtime):
        return datetime.fromtimestamp(unixtime)
    
    df['timeCreate'] = df['timeCreate'].apply(to_datetime)
    df.sort_values(by='timeCreate', inplace=True)
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    bins = pd.date_range(np.min(df['timeCreate'].values), np.max(df['timeCreate'].values), freq='600T') # по 10 минут

    df['cut'] = pd.cut(df['timeCreate'], bins, right=False)
    df = df.astype(str)
    df['cut'] = [x.replace('nan', str(bins[-1])) if x == 'nan' else x for x in df['cut'].values]
    df['cut'] = [x.split(',')[0].replace("[", '') for x in df['cut'].values]
    # df.loc[0, 'timeCreate'] = df.loc[0, 'timeCreate'] + timedelta(minutes=9)
    # df.loc[df.shape[0]-1, 'timeCreate'] = df.loc[df.shape[0]-1, 'timeCreate'] - timedelta(minutes=9)

    # мержинг данных на 10 минутки
    df_bins = pd.DataFrame(bins, columns=['cut']).astype(str).set_index('cut')
    df_bins['cut'] = list(df_bins.index)

    df = df_bins.set_index('cut').join(df.set_index('cut'))
    df.fillna('', inplace=True)

    df['timeCreate'] = list(df.index)
    df.reset_index(inplace=True)
    df.reset_index(inplace=True)
    df.drop(['index', 'cut'], axis=1, inplace=True)
    df = df[['hub', 'timeCreate', 'audienceCount']]

    df['audienceCount'] = [int(x) if x != '' else x for x in df['audienceCount'].values]
    listhubs = [x for x in list(set(df['hub'].values)) if x != '']
    set_timeCreate = set(df['timeCreate'].values)

    # добавляем не заполненные N-минутки по источнику данными по времени и 0 по аудитории (т.е. в этот период 10 мин не было сообщ)
    for i in range(len(listhubs)):
        
        df_ban = df[df['hub'] == listhubs[i]]
        # недостающие временные отрезки
        delta_set = set_timeCreate - set(df_ban['timeCreate'].values)
            
        if delta_set != set():
            df_need = pd.DataFrame(zip([listhubs[i]]*len(delta_set), delta_set, [0]*len(delta_set)))
            df_need.columns = ['hub', 'timeCreate', 'audienceCount']
            df = pd.concat([df, df_need], ignore_index=True)
        
        else:
            df_need = pd.DataFrame(zip([listhubs[i]]*len(set_timeCreate), set_timeCreate, [0]*len(set_timeCreate)))
            df_need.columns = ['hub', 'timeCreate', 'audienceCount']
            df = pd.concat([df, df_need], ignore_index=True)
        
    df.sort_values(by='timeCreate', inplace=True)

    # подготовка итогового словаря с hub и аудиторией
    hub_dcts = [df[df['hub'] == x][['timeCreate', 'audienceCount']].set_index('timeCreate').to_dict() for x in listhubs]

    for i in range(len(hub_dcts)):
        hub_dcts[i][listhubs[i]] = hub_dcts[i].pop('audienceCount')

    dynamicdata_audience = []
    for i in range(len(hub_dcts)):
        dynamicdata_audience.append({list(hub_dcts[i].keys())[0]:{str(int(time.mktime(datetime.strptime(key, "%Y-%m-%d %H:%M:%S").timetuple()))): str(val) for key, val in hub_dcts[i][list(hub_dcts[i].keys())[0]].items()}})

    # мин и макс даты в выбранном интервале времени (10 мни, 20 мин..)
    mind, maxd = list(dynamicdata_audience[0][list(dynamicdata_audience[0].keys())[0]].keys())[0], list(dynamicdata_audience[0][list(dynamicdata_audience[0].keys())[0]].keys())[-1]
    mind, maxd

    dynamicdata_audience = dict(ChainMap(*dynamicdata_audience))

    def sum_data(lst): # последовательно накапливает/суммирует кол-во по аудитории по столбцу..[1, 2, 4, 0, 2] -> [1, 3, 7, 7, 9..] 
        for i in range(len(lst)-1):
            lst[i+1] = lst[i] + lst[i+1]
        return lst

    for key in dynamicdata_audience.keys():
        dynamicdata_audience[key] = dict(zip([int(x[0]) for x in dynamicdata_audience[key].items()], [str(x) for x in sum_data([int(x[1]) for x in dynamicdata_audience[key].items()])]))

    values = ModelInfGraph(values=data, post=post, repost=repost, SMI=SMI, dynamicdata_audience=dynamicdata_audience)
    return  values


@app.get("/themes")
async def themes_analize(user: User = Depends(current_user), index: int =None, 
                             min_date=None, max_date=None) -> ThemesModel:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    os.chdir('/home/dev/fastapi/analytics_app/files')
    # данные с описанием тематик
    # filename = indexes[index] + '_LLM'
    os.chdir('/home/dev/fastapi/analytics_app/files/Росбанк/')
    filename = 'rosbank_01.04.2024-15.04.2024_LLM'
    with open (filename, 'rb') as fp:
        data = pickle.load(fp)


    data = [x[0]['generated_text'].split('model\n')[1] if len(x) == 1 else x for x in data]
    data = pd.DataFrame(data) 

    # print(data)

    query = {
            "size": 10000,
            "query": {
                        "range": {
                            "timeCreate": {      # skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024
                                "gte": min_date, # 1705329992
                                "lte": max_date, # 1705848392
                                "boost": 2.0
                            }
                        }
                    }
                }
    
    # данные с авторами, текстами и метаинформацией
    # dict_train = es.search(index='skillfactory_15.01.2024-21.01.2024', body=query)
    dict_train = es.search(index=indexes[index], body=query)
    dict_train = dict_train['hits']['hits']
    dict_train = [x['_source'] for x in dict_train]
    
    # with codecs.open(indexes[index], "r", "utf_8_sig") as train_file:
    #     dict_train = json.load(train_file)

    columns = ['timeCreate', 'text', 'hub', 'url', 'hubtype',
        'commentsCount', 'audienceCount',
        'citeIndex', 'repostsCount', 'likesCount', 'er', 'viewsCount',
        'toneMark', 'role',
        'country', 'region', 'city', 'language', 'fullname',
        'author_url', 'author_type', 'sex', 'age']

    author_df = pd.DataFrame(list(pd.DataFrame(dict_train)['authorObject'].values))
    author_df.columns=['fullname', 'author_url', 'author_type', 'sex', 'age']
    df_res = pd.DataFrame(dict_train).join(author_df)
    df_res = df_res[columns]
    # df_res.columns = ['Время', 'Текст', 'Источник', 'Ссылка', 'Тип источника', 'Комментариев', 'Аудитория',
    #        'Сайт-Индекс', 'Репостов', 'Лайков', 'Суммарная вовлеченность', 'Просмотров',
    #        'Тональность', 'Роль', 'Страна',
    #        'Регион', 'Город', 'Язык', 'Имя автора', 'Ссылка на автора', 'Тип автора',
    #        'Пол', 'Возраст']

    df_res = df_res.join(data)
    df_res = df_res[(df_res['timeCreate'] >= int(min_date)) & (df_res['timeCreate'] <= int(max_date))]
    df_res.reset_index(inplace=True)
    df_res.drop('index', axis=1, inplace=True)

    data = df_res[[0]]

    # функция для удаления лишних символов в текстах
    import re
    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    def words_only(text, regex=regex):
        try:
            return " ".join(regex.findall(text))
        except:
            return ""

    # удаляем лишние символы, оставляем слова
    data[0] = data[0].apply(words_only)

    # получение векторов текстов и сравнение
    count_vectorizer = CountVectorizer()
    vector_matrix = count_vectorizer.fit_transform(
        data[0].values)

    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    dff = pd.DataFrame(cosine_similarity_matrix)
    # dff = dff.round(5)
    # dff = dff.replace([1.000], 0)

    val_dff = dff.values
    # заменяем значения по главной диагонали на 0
    for i in range(len(val_dff)):
        val_dff[i][i] = 0
        
    dff = pd.DataFrame(val_dff)

    # создаем словарь похожих текстов вида {11: [12, 132],  44: [190], ...}
    fin_dict = {}
    threashhold = 0.70

    # print('threashhold')

    # выявляем список строк с похожими текстам
    for i in range(dff.shape[0]):
        if list(np.where(dff.loc[i].values >= threashhold)[0]) != []:
            if i not in [item for sublist in list(fin_dict.values()) for item in sublist]:

                fin_dict[i] = list(
                    np.where(dff.loc[i].values >= threashhold)[0])
                
        else:
            fin_dict[i] = []
            
    len_val = [len(x) for x in fin_dict.values()]
    dct_len_val = dict(zip(list(fin_dict.keys()), len_val))
    # dct_len_val = dict(sorted(dct_len_val.items(), key=itemgetter(1), reverse=True))

    # добавление текстов и метаданных в итоговый словарь
    fin_data = []
    texts = []
    texts_list = data.loc[list(fin_dict.keys())][0].values # список текстов с описанием, берется первое описание по первому тексту-ключу
    list_len = list(dct_len_val.values()) # список с количеством текстов по тематике
    # [{'description': 'Тема текста связана с ..', 'count': 152, 'texts': [...]},
    #  {'description': 'Тема текста связана с ..', 'count': 141, 'texts': [...]}, ..]

    for i in range(len(fin_dict.keys())):
        
        if fin_dict[list(fin_dict.keys())[i]] != []:

            a = {}
            a['description'] = texts_list[i] # описание тематики
            a['count'] = list_len[i] # количество текстов по тематике
            a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
            a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
            a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
            a['texts'] = 'texts' 
            # texts.append(df_res[df_res.index.isin(fin_dict[list(fin_dict.keys())[i]])].to_dict(orient='records'))
            fin_data.append(a)
            
        else:
            
            a = {}
            a['description'] = texts_list[i] # описание тематики
            a['count'] = list_len[i] # количество текстов по тематике
            a['audience'] = str(np.sum([x['audienceCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['audienceCount'] != ''])) # количество аудитории в тематике
            a['er'] = str(np.sum([x['er'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['er'] != ''])) # количество вовлеченности в тематику
            a['viewsCount'] = str(np.sum([x['viewsCount'] for x in df_res.iloc[fin_dict[list(fin_dict.keys())[i]]].to_dict(orient='records') if x['viewsCount'] != '']))# количество просмотров в тематике
            a['texts'] = 'texts'
            # texts.append(df_res.iloc[[list(fin_dict.keys())[i]]].to_dict(orient='records'))
            fin_data.append(a)
  
    return ThemesModel(values=fin_data)


@app.get("/voice")
async def voice_analize(user: User = Depends(current_user), index: int = None, 
                             min_date: int=None, max_date: int=None, query_str: str = None) -> ModelVoice:
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    search = query_str.split(',')
    topn = 20 # ТОП-источников, остальные пойдут в "Другие"
    values = []

    for i in range(len(search)):

        data = elastic_query(theme_index=indexes[index], query_str=search[i])
        print(len(data))
        # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', query_str='data')

        # отфильтровываем по необходимой дате из календаря
        data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
        
    #     data = elastiqsearc(search[i]) # данные из эластик
        search_name = search[i].strip()
        hubs_tonality = Counter([(x['hub'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')) for x in data])
        list_tonal_hubs = [[key[0], key[1], val] for key, val in hubs_tonality.items()]

        lst_dicts = [{x[0]: {x[1]: x[2]}} for x in list_tonal_hubs] # {'youtube.com': {'Нейтрал': 2}}, {'yaroslavl.bezformata.com': {'Нейтрал': 1}},
        keys_list = list(set([list(x.keys())[0] for x in lst_dicts]))

        hubs_tonality_dict = {} # финальный словарь по источникуам и тональности

        for j in range(len(keys_list)):
            list_same_dict = [x for x in lst_dicts if keys_list[j] in x]
            
            if len(list_same_dict) != 1:
            
                dict_hub_ton = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]] = {}

                for i in range(len(list_same_dict)):
                    dict_hub_ton[list(list_same_dict[0].keys())[0]].update(list(list_same_dict[i].values())[0])
                    
                hubs_tonality_dict.update(dict_hub_ton)
                    
            else:
                dict_hub_ton = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]] = {}
                dict_hub_ton[list(list_same_dict[0].keys())[0]].update(list(list_same_dict[0].values())[0])
                
                hubs_tonality_dict.update(dict_hub_ton)

        sort = Counter(dict(zip(list(hubs_tonality_dict.keys()), [np.sum(list(x.values())) for x in list(hubs_tonality_dict.values())]))).most_common()
        sort = [x[0] for x in sort]

        # финальная сортировка по количеству
        index_map = {v: i for i, v in enumerate(sort)}
        hubs_tonality_dict = sorted(hubs_tonality_dict.items(), key=lambda pair: index_map[pair[0]])

        hubs_tonality_dict = [{x[0]: x[1]} for x in hubs_tonality_dict]
        # hubs_tonality_dict = [{'source': x} for x in hubs_tonality_dict]
        dcts = [{'source': list(x.keys())[0]} for x in hubs_tonality_dict] # {'source': 'vk.com'}

        for i in range(len(dcts)):
            dcts[i].update([list(x.values())[0] for x in hubs_tonality_dict][i]) # {'source': 'vk.com', 'Нейтрал': 29, 'Негатив': 5}

        # [{'source': 'vk.com', 'Нейтрал': 29, 'Негатив': 5, 'Позитив': 0}, ...
        for i in range(len(dcts)):
            if 'Нейтрал' not in dcts[i]:
                dcts[i]['Нейтрал'] = 0
            if 'Позитив' not in dcts[i]:
                dcts[i]['Позитив'] = 0
            if 'Негатив' not in dcts[i]:
                dcts[i]['Негатив'] = 0


        ##### источники - тональность - тип сообщения
        hubs = Counter([x['hub'] for x in data])
        hubs = dict(sorted(hubs.items(), key=lambda x: x[1], reverse=True)[:topn])

        list_topn_hubs = list(hubs.keys())
        message_tonality = [[x['hub'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')] 
                            for x in data if x['hub'] in list_topn_hubs]


        message_tonality_type = [[x['hub'], x['type'], str(x['toneMark']).replace('0', 'Нейтрал').replace('-1', 'Негатив').replace('1', 'Позитив')] 
                            for x in data if x['hub'] in list_topn_hubs]

        dct_tonality_hubs = Counter([', '.join(x) for x in message_tonality_type])

        hub_tonality_type_list = [[x[0].split(',')[0].strip(), x[0].split(',')[1].strip(), x[0].split(',')[2].strip(), 
                            x[1]] for x in list(dct_tonality_hubs.items())]
        hub_tonality_type_list = sorted(hub_tonality_type_list, key=itemgetter(3), reverse=True)
        
        for i in range(len(hub_tonality_type_list)):
            data = hub_tonality_type_list[i]
            data.append(search_name)
            hub_tonality_type_list[i] = dict(zip(["hub", "type", "tonality", "count", "search"], data))
        
        values_search = {}
        values_search['name'] = search_name
        values_search['tonality'] = dcts
        values_search['sunkey_data'] = hub_tonality_type_list

        values.append(values_search)

    return ModelVoice(__root__ = values)


@app.get("/media-rating")
def media_rating(user: User = Depends(current_user), index: int = None, min_date: int=None,  
                 max_date: int=None) -> MediaRatingModel:
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str='all')
    # data = es.search(index='skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024', query_str='data')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    df = pd.DataFrame(data)

    # метаданные
    # разбивка и сборка соцмедиа и СМИ в один датафрэйм с данными
    df_meta = pd.DataFrame()

    # случай выгрузки темы только по СМИ
    if 'hubtype' not in df.columns:

        dff = df
        dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
            '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
        df_meta_smi_only = dff[[
            'timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
        # df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text', 'citeIndex']
        df_meta_smi_only['fullname'] = dff['hub']
        df_meta_smi_only['author_type'] = 'Онлайн-СМИ'
        df_meta_smi_only['hubtype'] = 'Онлайн-СМИ'
        df_meta_smi_only['type'] = 'Онлайн-СМИ'
        df_meta_smi_only['er'] = 0
        df_meta_smi_only.dropna(subset=['timeCreate'], inplace=True)
        df_meta_smi_only = df_meta_smi_only.set_index(['timeCreate'])
        df_meta_smi_only['date'] = [x[:10] for x in df_meta_smi_only.index]
    #     df_meta_smi_only = df_meta_smi_only[columns]

        df_meta = df_meta_smi_only

    if 'hubtype' in df.columns:

        for i in range(2):  # Онлайн-СМИ или соцмедиа

            if i == 0:
                dff = df[df['hubtype'] != 'Онлайн-СМИ']
                if dff.shape[0] != 0:

                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
                        '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_socm = dff[['timeCreate', 'hub', 'toneMark',
                                        'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
                    df_meta_socm['fullname'] = pd.DataFrame.from_records(
                        dff['authorObject'].values)['fullname'].values
                    df_meta_socm['author_type'] = pd.DataFrame.from_records(
                        dff['authorObject'].values)['author_type'].values
                    df_meta_socm.dropna(
                        subset=['timeCreate'], inplace=True)
                    df_meta_socm = df_meta_socm.set_index(['timeCreate'])
                    df_meta_socm['date'] = [x[:10]
                                            for x in df_meta_socm.index]

            if i == 1:
                dff = df[df['hubtype'] == 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
                        '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_smi = dff[['timeCreate', 'hub', 'toneMark',
                                        'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                    df_meta_smi['fullname'] = dff['hub']
                    df_meta_smi['author_type'] = 'Онлайн-СМИ'
                    df_meta_smi['hubtype'] = 'Онлайн-СМИ'
                    df_meta_smi['type'] = 'Онлайн-СМИ'
                    df_meta_smi.dropna(subset=['timeCreate'], inplace=True)
                    df_meta_smi = df_meta_smi.set_index(['timeCreate'])
                    df_meta_smi['date'] = [x[:10]
                                            for x in df_meta_smi.index]

        if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
            df_meta = pd.concat([df_meta_socm, df_meta_smi])
        elif 'df_meta_smi' and 'df_meta_socm' not in locals():
            df_meta = df_meta_smi
        else:
            df_meta = df_meta_socm


    if set(df_meta['hub'].values) == {"telegram.org"}:

        df_meta = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (
            df_meta['hub'] == "telegram.org")]

        # negative smi
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)][
            ['fullname', 'audienceCount']].values

        dict_neg = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_neg.keys():

                dict_neg[df_hub_siteIndex[i][0]] = []
                dict_neg[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

            else:
                dict_neg[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

        list_neg = [list(set(x)) for x in dict_neg.values()]
        list_neg = [[0] if x[0] ==
                    'n/a' else x for x in list_neg if x != 'n/a']
        list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]

        for i in range(len(list_neg)):
            dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

        dict_neg = dict(
            sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

        dict_neg_hubs_count = dict(
            Counter(list(
                df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == -1)]['fullname'])))

        fin_neg_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_neg, dict_neg_hubs_count):
            for key, value in d.items():
                fin_neg_dict[key] += (value,)

        list_neg_smi = list(fin_neg_dict.keys())
        list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
        list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

        # positive smi
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)][
            ['fullname', 'audienceCount']].values

        dict_pos = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_pos.keys():

                dict_pos[df_hub_siteIndex[i][0]] = []
                dict_pos[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

            else:
                dict_pos[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

        list_pos = [list(set(x)) for x in dict_pos.values()]
        list_pos = [[0] if x[0] ==
                    'n/a' else x for x in list_pos if x != 'n/a']
        list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]

        for i in range(len(list_pos)):
            dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

        dict_pos = dict(
            sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

        dict_pos_hubs_count = dict(
            Counter(list(
                df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] == 1)]['fullname'])))

        fin_pos_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_pos, dict_pos_hubs_count):
            for key, value in d.items():
                fin_pos_dict[key] += (value,)

        list_pos_smi = list(fin_pos_dict.keys())
        list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
        list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]

        # data to bobble graph
        df_meta['timeCreate'] = list(df_meta.index)
        
        bobble = []
        df_tonality = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
            ['fullname', 'audienceCount', 'toneMark', 'url']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Мессенджеры каналы') & (df_meta['toneMark'] != 0)][
            ['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1,
                                                                                                1)).total_seconds() * 1000)
                    for x in date_ton]

        for i in range(len(df_tonality)):
            if df_tonality[i][2] == -1:
                bobble.append([date_ton[i], df_tonality[i][0],
                                dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
            elif df_tonality[i][2] == 1:
                bobble.append([date_ton[i], df_tonality[i][0],
                                dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])

        for i in range(len(bobble)):
            if bobble[i][3] == 1:
                bobble[i][3] = "#32ff32"
            else:
                bobble[i][3] = "#FF3232"


        data = {
            "neg_smi_name": list_neg_smi,
            "neg_smi_count": list_pos_smi_massage_count,
            "neg_smi_rating": list_neg_smi_index,
            "pos_smi_name": list_pos_smi,
            "pos_smi_count": list_pos_smi_massage_count,
            "pos_smi_rating": list_pos_smi_index,

            "date_bobble": [x[0] for x in bobble],
            "name_bobble": [x[1] for x in bobble],
            "index_bobble": [x[2] for x in bobble],
            "z_index_bobble": [1] * len(bobble),
            "tonality_index_bobble": [x[3] for x in bobble],
            "tonality_url": [x[4] for x in bobble],
        }

        return data

    df_meta = df_meta[df_meta['hubtype'] == 'Онлайн-СМИ']

    # negative smi
    df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)][
        ['hub', 'citeIndex']].values

    dict_neg = {}
    for i in range(len(df_hub_siteIndex)):

        if df_hub_siteIndex[i][0] not in dict_neg.keys():

            dict_neg[df_hub_siteIndex[i][0]] = []
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

        else:
            dict_neg[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

    list_neg = [list(set(x)) for x in dict_neg.values()]
    list_neg = [[0] if x[0] ==
                'n/a' else x for x in list_neg if x != 'n/a']
    list_neg = [int(x[0]) if x[0] != '' else 0 for x in list_neg]

    for i in range(len(list_neg)):
        dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

    dict_neg = dict(
        sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

    dict_neg_hubs_count = dict(
        Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)]['hub'])))

    fin_neg_dict = defaultdict(tuple)
    # you can list as many input dicts as you want here
    for d in (dict_neg, dict_neg_hubs_count):
        for key, value in d.items():
            fin_neg_dict[key] += (value,)

    list_neg_smi = list(fin_neg_dict.keys())
    list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
    list_neg_smi_massage_count = [x[1] for x in fin_neg_dict.values()]

    # positive smi
    df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)][
        ['hub', 'citeIndex']].values

    dict_pos = {}
    for i in range(len(df_hub_siteIndex)):

        if df_hub_siteIndex[i][0] not in dict_pos.keys():

            dict_pos[df_hub_siteIndex[i][0]] = []
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

        else:
            dict_pos[df_hub_siteIndex[i][0]].append(df_hub_siteIndex[i][1])

    list_pos = [list(set(x)) for x in dict_pos.values()]
    list_pos = [[0] if x[0] ==
                'n/a' else x for x in list_pos if x != 'n/a']
    list_pos = [int(x[0]) if x[0] != '' else 0 for x in list_pos]

    for i in range(len(list_pos)):
        dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

    dict_pos = dict(
        sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

    dict_pos_hubs_count = dict(
        Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)]['hub'])))

    fin_pos_dict = defaultdict(tuple)
    # you can list as many input dicts as you want here
    for d in (dict_pos, dict_pos_hubs_count):
        for key, value in d.items():
            fin_pos_dict[key] += (value,)

    list_pos_smi = list(fin_pos_dict.keys())
    list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
    list_pos_smi_massage_count = [x[1] for x in fin_pos_dict.values()]


    df_meta['timeCreate'] = list(df_meta.index)

    # data to bobble graph
    bobble = []
    df_tonality = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
        ['hub', 'citeIndex', 'toneMark', 'url']].values
    index_ton = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
        ['timeCreate']].values.tolist()
    date_ton = [x[0] for x in index_ton]
    date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1, 1)).total_seconds() * 1000)
                for x in date_ton]

    for i in range(len(df_tonality)):
        if df_tonality[i][2] == -1:
            bobble.append([date_ton[i], df_tonality[i][0],
                            dict_neg[df_tonality[i][0]], -1, df_tonality[i][3]])
        elif df_tonality[i][2] == 1:
            bobble.append([date_ton[i], df_tonality[i][0],
                            dict_pos[df_tonality[i][0]], 1, df_tonality[i][3]])

    for i in range(len(bobble)):
        if bobble[i][3] == 1:
            bobble[i][3] = "#32ff32"
        else:
            bobble[i][3] = "#FF3232"

    values = {}
    values['first_graph'] = {}
    values['first_graph']['negative_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_neg_smi, list_neg_smi_index, list_neg_smi_massage_count)]
    values['first_graph']['positive_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_pos_smi, list_pos_smi_index, list_pos_smi_massage_count)]

    values['second_graph'] = ''
    values['second_graph'] = [{'name': x, 'time': y, 'index': z, 'url': u,'color': t} for (x,y,z,u,t) in zip([x[1] for x in bobble], [x[0] for x in bobble], [x[2] for x in bobble], [x[4] for x in bobble], [x[3] for x in bobble])]

    return MediaRatingModel(first_graph=values['first_graph'], second_graph=values['second_graph'])


@app.get('/ai-analytics')
async def ai_analytics_get(index: int=None, min_date: int=None, max_date: int=None) -> ModelAiAnalytics: 
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str='all')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    keys = ['id', 'timeCreate', 'text', 'hub', 'audienceCount', 'commentsCount', 'er', 'url'] # ключи для отображения в первой таблице
    data = [{k: y.get(k, None) for k in keys} for y in data[:100]] # данные для первой таблицы
    ranges = list(np.arange(0, len(data)))
    [x.update({'id': y.item()}) for x, y in zip(data, ranges)] # меняем значение id на 0,1,2...для передачи далее при выборе на LLM

    return ModelAiAnalytics(data=data)


# async def create_llm_query(data, query, task_id, text_ids):

#     LLM = {} # данные для возврата вида {'text': 'описание'}
#     LLM['promt'] = query
#     LLM['texts'] = []

#     print(task_id)
#     # print(redis_db[task_id], "step")
#     # print(len(data))

#     for i in range(len(data)): # цикл работы LLM с выбранными текстами

#         from transformers import AutoTokenizer, pipeline
#         os.chdir('/home/dev/fastapi/analytics_app/data/LLM_models')

#         model = "gemma-2b-it"
#         tokenizer = AutoTokenizer.from_pretrained(model)
#         pipeline = pipeline(
#             "text-generation",
#             model=model,
#             model_kwargs={"torch_dtype": torch.bfloat16},
#             device="cuda",
#         ) 
        
#         st = time.time()      

#         # print(int(((i + 1) / len(data)*100)))

#         messages = [
#             {"role": "user", "content": query + data[i]['text']},
#         ]
#         prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         outputs = pipeline( 
#             prompt,
#             max_new_tokens=256,
#             do_sample=True,
#             temperature=0.7,
#             top_k=50,
#             top_p=0.95,
#             batch_size=32
#         )
#         LLM['texts'].append({'id':text_ids[i], 'text': data[i]['text'], 
#                                 'llm_text': outputs[0]['generated_text'].split('model\n')[1]})

#         torch.cuda.empty_cache()
#         gc.collect()

#         del prompt
#         del outputs
#         del model 
#         del pipeline

#         redis_db[task_id] = int(((i + 1) / len(data)*100))

#         # get the execution time
#         et = time.time()
#         elapsed_time = et - st 
#         # print(elapsed_time)
#         # print('Execution time:', elapsed_time, 'seconds') 

#         await asyncio.sleep(1)
#         redis_db[task_id] = int(((i + 1) / len(data)*100))
    

# @app.post('/ai-analytics')
# async def ai_analytics_post(query: QueryAiLLM):

#     # Путь к файлу с темами 
#     file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
#     # Загрузка словаря с темами
#     indexes = load_dict_from_pickle(file_path)

#     # делаем запрос на текстовый поиск
#     data = elastic_query(theme_index=indexes[query.index], query_str='all')
#     # отфильтровываем по необходимой дате из календаря
#     data = [x for x in data if query.min_date <= x['timeCreate'] <= query.max_date]
    
#     # если был введен промт и выбраны строчки текстов (индексты в таблице данных), то начать запрос к LLM
#     # query.promt = "Какая тематика у этого текста? Текст: "
#     data = [data[x] for x in query.texts_ids]

#     # print(data)
#     task_id = "llm_task" + '_' + str(len(redis_db.keys()))
#     print(task_id)

#     redis_db.mset({task_id: 0})

#     task = asyncio.create_task(
#         create_llm_query(data, query.promt, task_id, query.texts_ids)
#     )

#     return task_id
    

# Определение модели запроса
class QueryCompetitors(BaseModel):
    themes_ind: list
    min_date: int
    max_date: int


class ValueCompetitor(BaseModel):
    timestamp: int
    count: int


class FirstGraphCompetitor(BaseModel):
    index_name: str
    values: List[ValueCompetitor]


class NegItem(BaseModel):
    hub: str
    count: int
    rating: int
    url: str


class Po(BaseModel):
    hub: str
    count: int
    rating: Union[int, str]
    url: str


class SMICompetitor(BaseModel):
    neg: List[NegItem]
    pos: List[Po]


class Po1(BaseModel):
    hub: str
    count: int
    rating: int
    url: str


class SocmediaCompetitor(BaseModel):
    neg: List[NegItem]
    pos: List[Po1]


class SecondGraphCompetitor(BaseModel):
    index_name: str
    SMI: SMICompetitor
    Socmedia: SocmediaCompetitor


class SMIItem(BaseModel):
    name: str
    count: int
    rating: Union[int, str]
    url: str


class SocmediaItem(BaseModel):
    name: str
    count: int
    rating: int
    url: str


class ThirdGraphCompetitor(BaseModel):
    index_name: str
    SMI: List[SMIItem]
    Socmedia: List[SocmediaItem]


class CompetitorsModel(BaseModel):
    first_graph: List[FirstGraphCompetitor]
    second_graph: List[SecondGraphCompetitor]
    third_graph: List[ThirdGraphCompetitor]


@app.post('/competitors', response_model=CompetitorsModel)
async def competitors(query: QueryCompetitors):
    # Путь к файлу с темами
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    another_graph = []
    min_date = []
    max_date = []
    themes_ind = query.themes_ind

    # Обработка данных для каждого theme_ind
    for i in range(len(themes_ind)):
        data = elastic_query(theme_index=indexes[themes_ind[i]], query_str='all')
        ind_df = [x for x in data if query.min_date <= x['timeCreate'] <= query.max_date]

        # Замена audience на audienceCount
        ind_df = [{"audienceCount" if k == "audience" else k: v for k, v in x.items()} for x in ind_df]

        # Формирование цензури для SMI
        for item in ind_df:
            if item['hubtype'] == 'Онлайн-СМИ':
                item['rating'] = item.get('citeIndex', 0)
            else:
                item['rating'] = item.get('audienceCount', 0)

        min_date.append(np.min([x['timeCreate'] for x in ind_df]))
        max_date.append(np.max([x['timeCreate'] for x in ind_df]))
        another_graph.append(ind_df)

    # Получение общей мин и макс даты
    dates = [min_date, max_date]
    min_date = np.min(dates[0])
    max_date = np.max(dates[1])
    filenames = [indexes[x] for x in themes_ind]

    # Формирование первого графика
    first_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)
        df['timeCreate'] = pd.to_datetime(df['timeCreate'], unit='s')
        min_date_dt = pd.to_datetime(min_date, unit='s')
        max_date_dt = pd.to_datetime(max_date, unit='s')
        df['bins'] = pd.cut(df['timeCreate'], pd.date_range(min_date_dt, max_date_dt, freq='30T'))
        aggregated_data = df.groupby('bins').size().reset_index(name='count')
        aggregated_data['time'] = aggregated_data['bins'].apply(lambda x: x.left.timestamp())

        first_graph.append({
            'index_name': filename,
            'values': [{'timestamp': int(row.time * 1000), 'count': row.count} for row in aggregated_data.itertuples()]
        })

    # Формирование второго графика (second_graph)
    second_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)

        # Данные только по SMI (hubtype == 'Онлайн-СМИ')
        smi_data = df[df['hubtype'] == 'Онлайн-СМИ']
        neg_smi = smi_data[smi_data['toneMark'] == -1].groupby('hub').agg(
            count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()

        pos_smi = smi_data[smi_data['toneMark'] == 1].groupby('hub').agg(
            count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()

        second_graph.append({
            'index_name': filename,
            'SMI': {
                'neg': [{'hub': row['hub'], 'count': row['count'], 'rating': row['citeIndex'], 'url': row['url']} for _, row in neg_smi.iterrows()],
                'pos': [{'hub': row['hub'], 'count': row['count'], 'rating': row['citeIndex'], 'url': row['url']} for _, row in pos_smi.iterrows()],
            }
        })

        # Данные только по Соцмедиа (hubtype != 'Онлайн-СМИ')
        socmedia_data = df[df['hubtype'] != 'Онлайн-СМИ']
        neg_socmedia = socmedia_data[socmedia_data['toneMark'] == -1].groupby('hub').agg(
            count=('hub', 'size'),
            audienceCount=('audienceCount', 'first'),
            url=('url', 'first')
        ).reset_index()

        pos_socmedia = socmedia_data[socmedia_data['toneMark'] == 1].groupby('hub').agg(
            count=('hub', 'size'),
            audienceCount=('audienceCount', 'first'),
            url=('url', 'first')
        ).reset_index()

        second_graph[-1]['Socmedia'] = {
            'neg': [{'hub': row['hub'], 'count': row['count'], 'rating': row['audienceCount'], 'url': row['url']} for _, row in neg_socmedia.iterrows()],
            'pos': [{'hub': row['hub'], 'count': row['count'], 'rating': row['audienceCount'], 'url': row['url']} for _, row in pos_socmedia.iterrows()],
        }

    # Формирование третьего графика (third_graph)
    third_graph = []
    for theme_data, filename in zip(another_graph, filenames):
        df = pd.DataFrame(theme_data)

        # SMI данные
        df_smi = df[df['hubtype'] == 'Онлайн-СМИ']
        smi_data = df_smi.groupby('hub').agg(
            hub_count=('hub', 'size'),
            citeIndex=('citeIndex', 'first'),
            url=('url', 'first')
        ).reset_index()

        smi_results = [{
            'name': row['hub'],
            'count': row['hub_count'],
            'rating': row['citeIndex'],
            'url': row['url']
        } for _, row in smi_data.iterrows()]

        # Socmedia данные
        df_socmedia = df[df['hubtype'] != 'Онлайн-СМИ']
        socmedia_data = df_socmedia.groupby('hub').agg(
            hub_count=('hub', 'size'),
            audienceCount=('audienceCount', 'first'),
            url=('url', 'first')
        ).reset_index()

        socmedia_results = [{
            'name': row['hub'],
            'count': row['hub_count'],
            'rating': row['audienceCount'],
            'url': row['url']
        } for _, row in socmedia_data.iterrows()]

        third_graph.append({
            'index_name': filename,
            'SMI': smi_results,
            'Socmedia': socmedia_results,
        })

    return {
        'first_graph': first_graph,
        'second_graph': second_graph,
        'third_graph': third_graph,
    }


# @app.get('/data-folders')
# async def data_folders(user: User = Depends(current_user)) -> ModelDataFolder:

#     es_indexes = [index for index in es.indices.get('*')] # список всех индексов elastic
#     es_indexes = [x.strip() for x in es_indexes]

#     if user.theme_rules["perm"] == 'admin': # если пользователь админ, то вернуть все темы

#         folders = '/home/dev/fastapi/analytics_app/data/json_files'
#         sub_folders = [name for name in os.listdir(folders) if os.path.isdir(os.path.join(folders, name))]

#         data_values = []
#         os.chdir(folders)
#         for i in range(len(sub_folders)):
#             data_values.append({"name": sub_folders[i], 
#                                "values": [f for f in listdir(sub_folders[i]) if isfile(join(sub_folders[i], f))]}) 
      
#         return ModelDataFolder(values=data_values)
    
#     else: # если пользователь не админ, то вернуть его темы
#         data_index = []
#         user_index = list(set(es_indexes) & set([x.strip().lower().replace('.json', '') for x in user.theme_rules.split(',')]))

#         return ModelDataFolder(values=data_values)


@app.get("/create-data-projector/{user_id}/{folder_name}/{file_name}")
async def create_data_projector(user_id: str, folder_name: str, file_name: str):
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    indexes = load_dict_from_pickle(file_path)

    embed = hub.load("/home/dev/fastapi/analytics_app/data/embed_files/universal-sentence-encoder-multilingual_3")

    # Полный путь к файлу
    file_path = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/{file_name}' + '.json'

    try:
        with io.open(file_path, encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file, strict=False)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при чтении JSON: {e}")
        
        a = []
        try:
            with open(file_path, encoding='utf-8', mode='r') as file:
                for line in file:
                    a.append(line)

            dict_train = []
            for i in range(len(a)):
                try:
                    dict_train.append(ast.literal_eval(a[i]))
                except (SyntaxError, ValueError):
                    continue
            dict_train = [x[0] for x in dict_train]

        except FileNotFoundError: 
            raise HTTPException(status_code=404, detail="File not found")

    df = pd.DataFrame(dict_train)
    df_meta = pd.DataFrame()

    if 'hubtype' not in df.columns:
        dff = df
        dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
        df_meta_smi_only = dff[['timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
        df_meta_smi_only['fullname'] = dff['hub']
        df_meta_smi_only['author_type'] = 'Онлайн-СМИ'
        df_meta_smi_only['hubtype'] = 'Онлайн-СМИ'
        df_meta_smi_only['type'] = 'Онлайн-СМИ'
        df_meta_smi_only['er'] = 0
        df_meta = df_meta_smi_only

    if 'hubtype' in df.columns:
        for i in range(2):
            if i == 0:
                dff = df[df['hubtype'] != 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_socm = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
                    df_meta_socm['fullname'] = pd.DataFrame.from_records(dff['authorObject'].values)['fullname'].values
                    df_meta_socm['author_type'] = pd.DataFrame.from_records(dff['authorObject'].values)['author_type'].values

            if i == 1:
                dff = df[df['hubtype'] == 'Онлайн-СМИ']
                if dff.shape[0] != 0:
                    dff['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
                    df_meta_smi = dff[['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
                    df_meta_smi['fullname'] = dff['hub']
                    df_meta_smi['author_type'] = 'Онлайн-СМИ'
                    df_meta_smi['hubtype'] = 'Онлайн-СМИ'
                    df_meta_smi['type'] = 'Онлайн-СМИ'

        if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
            df_meta = pd.concat([df_meta_socm, df_meta_smi])
        elif 'df_meta_smi' and 'df_meta_socm' not in locals():
            df_meta = df_meta_smi
        else:
            df_meta = df_meta_socm

    df_text = df_meta[['text']]
    
    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    def words_only(text, regex=regex):
        try:
            return " ".join(regex.findall(text))
        except:
            return ""

    mystopwords = ['это', 'наш', 'тыс', 'млн', 'млрд', 'также', 'т', 'д', 'URL',
                   'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
                   ' - ', '-', 'В', '—', '–', '-', 'в', 'который']

    def preprocess_text(text):
        text = text.lower().replace("ё", "е")
        text = re.sub('((www\[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()

    def remove_stopwords(text, mystopwords=mystopwords):
        try:
            return " ".join([token for token in text.split() if not token in mystopwords])
        except:
            return ""

    df_text['text'] = df_text['text'].apply(words_only)
    df_text['text'] = df_text['text'].apply(preprocess_text)
    df_text['text'] = df_text['text'].apply(remove_stopwords)
    df_text = df_text[df_text['text'].notna()]
    df_text = df_text[df_text['text'] != '']

    sent_ru = df_text['text'].values
    sent_ru = sent_ru[:50]

    # Обработка по партиям
    batch_size = 8
    embeddings = []
    for i in range(0, len(sent_ru), batch_size):
        batch = sent_ru[i:i + batch_size]
        with tf.device('/CPU:0'):
            embeddings.append(embed(batch))

    # Объединение эмбеддингов в один массив
    embeddings = tf.concat(embeddings, axis=0)

    embed_list = embeddings

    dff = pd.DataFrame(embeddings)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(pd.DataFrame(embed_list).values)

    coord_list = [', '.join(map(str, x)) for x in x_tsne.tolist()]
    names_list = [re.sub('\n', ' ', name) for name in df_meta['fullname'].fillna('None').values.tolist()]

    # Создание директории для сохранения файлов, если она не существует
    project_files_dir = f'/home/dev/fastapi/analytics_app/data/{user_id}/projector_files_directory/{folder_name}/'
    os.makedirs(project_files_dir, exist_ok=True)

    # сохранение данных для tsne
    dict_tsne = {
        'author_name_str': '\n'.join(names_list),
        'coord_list_str': '\n'.join(coord_list)
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    tsv_filename = f"{file_name}_authors_point_{timestamp}.tsv"
    txt_filename = f"{file_name}_authors_name_{timestamp}.txt"

    # Сохранение файлов
    try:
        # Сохранение tsv файла
        with open(os.path.join(project_files_dir, tsv_filename), 'w') as f:
            for line in embed_list:
                f.write('\t'.join(map(str, line)) + '\n')

        # Сохранение txt файла
        with open(os.path.join(project_files_dir, txt_filename), 'w', encoding='utf-8') as f:
            for line in names_list:
                f.write(line + '\n')
    except Exception as e:
        print(f"Ошибка при сохранении файлов: {e}")

    # Сохранение данных о папке и файлах в Redis
    user_data = redis_db.hgetall(user_id)

    if not user_data:  # Проверяем, есть ли данные
        raise Exception("User data does not exist.")

    # Проверяем, существует ли field для projector_files_directory
    if "projector_files_directory" in user_data:
        user_folders = json.loads(user_data["projector_files_directory"])
    else:
        user_folders = {}  # Инициализируем пустой словарь, если поле отсутствует

    # Добавляем информацию о новых файлах в соответствующую папку
    file_info = {
        "tsv-file": tsv_filename,
        "txt-file": txt_filename,
        "creation_date": timestamp
    }

    # Добавляем новый файл в соответствующую папку
    if folder_name not in user_folders:
        user_folders[folder_name] = []

    user_folders[folder_name].append(file_info)

    # Сохраняем обновленные данные обратно в Redis
    redis_db.hset(user_id, "projector_files_directory", json.dumps(user_folders))

    return f"Файлы авторов для прожектора темы {file_name} созданы и сохранены в папку {folder_name}!"


# @app.get("/projector-files")
# async def projector_files() -> ModelDataFolder:

#     folders = '/home/dev/fastapi/analytics_app/data/projector_files'
#     sub_folders = [name for name in os.listdir(folders) if os.path.isdir(os.path.join(folders, name))]

#     data_values = []
#     os.chdir(folders)
#     for i in range(len(sub_folders)):
#         data_values.append({"name": sub_folders[i], 
#                             "values": [f for f in listdir(sub_folders[i]) if isfile(join(sub_folders[i], f))]}) 
    
#     return ModelDataFolder(values=data_values)


@app.get('/file-load/{user_id}/{file_type}/{folder_name}/{file_name}')
def load_file(user_id: str, file_type: str, folder_name: str, file_name: str):

    # Основная директория, где хранятся папки с файлами
    BASE_DIR = '/home/dev/fastapi/analytics_app/data'
    PROJECTOR_DIR = os.path.join(BASE_DIR, user_id, 'projector_files_directory', folder_name)
    JSON_DIR = os.path.join(BASE_DIR, user_id, 'json_files_directory', folder_name)
    BERTOPIC_DIR = os.path.join(BASE_DIR, user_id, 'bertopic_files_directory', folder_name)

    # Определяем полный путь к файлу на основе типа файла
    if file_type == 'projector_files_directory':
        file_path = os.path.join(PROJECTOR_DIR, file_name)
    if file_type == 'bertopic_files_directory':
        file_path = os.path.join(BERTOPIC_DIR, file_name)
        print(file_path)
    elif file_type == 'json_files_directory':
        if '.json' not in file_name:
            file_name = file_name + '.json'
        file_path = os.path.join(JSON_DIR, file_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Use 'projector' or 'json'.")
    
    print(file_path)
    # Проверка существования файла
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Возврат файла
    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)

######################################## Запросы к LLM моделям #######################################

def update_task_progress(task_key, progress, queries):
    # Здесь queries - это список задач для данного пользователя
    for query in queries:
        # Проверяем, есть ли ключ задачи в текущем словаре
        if task_key in query:
            # Обновляем данные о прогрессе
            query[task_key] = {**query[task_key], **progress}
            return queries  # Возвращаем обновленные данные по мере нахождения задачи
    
    # Если задача не найдена, возвращаем исходные данные без изменений
    return queries

def update_progress(user_id, task_id, progress):
    os.chdir('/home/dev/fastapi/analytics_app/data')
    
    # Получаем текущую дату
    current_date = datetime.now().date().strftime('%Y-%m-%d')

    with open('llm_history_progress.pickle', 'rb') as file:
        llm_history = pickle.load(file)

    # Обновляем прогресс только для пользователя с соответствующим user_id
    for entry in llm_history:
        if entry['user_id'] == user_id:
            values = entry['values']
            date_queries = values.get('llm_queries', {})
            
            # Проверяем, есть ли у данного пользователя данные для текущей даты
            if isinstance(date_queries, dict):
                # Проверяем наличие задач для текущей даты
                if current_date in date_queries:
                    queries_for_date = date_queries[current_date]
                    updated_queries = update_task_progress(task_id, progress, queries_for_date)
                    date_queries[current_date] = updated_queries  # Обновляем список с задачами
            elif isinstance(date_queries, list):
                updated_queries = update_task_progress(task_id, progress, date_queries)
                values['llm_queries'] = updated_queries  # Обновляем данные

    # Сохраняем обновленные данные в файл
    with open('llm_history_progress.pickle', 'wb') as file:
        pickle.dump(llm_history, file)

# Модель для задачи
class AnalysisRequest(BaseModel):
    index: int = None
    min_date: int = None
    max_date: int = None
    query_str: str = None
    system_prompt: str = None
    example_promt: str = None
    main_prompt: str = None
    user_id: int = None
    folder_name: str = None


# Путь к файлу истории
HISTORY_FILE = '/home/dev/fastapi/analytics_app/data/llm_history_progress.pickle'

def load_history(user_id):
    """Загружает историю выполнения задач пользователя из файла."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as file:
            history = pickle.load(file)
            # Ищем запись для указанного user_id
            for entry in history:
                if entry['user_id'] == user_id:
                    return entry['values']
    return {}

def save_history(user_id, history_data):
    """Сохраняет данные о задачах пользователя в файл."""
    # Загружаем полную историю для обновления
    all_history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'rb') as file:
            all_history = pickle.load(file)

    # Смотрим, существует ли запись для данного user_id
    user_found = False
    for entry in all_history:
        if entry['user_id'] == user_id:
            entry['values'].update(history_data)
            user_found = True
            break

    if not user_found:
        # Если пользователь не найден, добавляем новый
        all_history.append({'user_id': user_id, 'values': history_data})

    # Сохраняем обновленную историю обратно в файл
    with open(HISTORY_FILE, 'wb') as file: 
        pickle.dump(all_history, file)


@app.post("/llm-analyze/")
async def llm_analyze(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Эндпойнт для анализа запросов LLM и сохранения их истории."""
    # Загружаем историю для пользователя
    user_history = load_history(request.user_id)
    current_date = datetime.now().date()
    print(current_date)

    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)

    # Определяем уникальный номер задач для текущей даты
    if 'llm_queries' not in user_history:
        user_history['llm_queries'] = {}

    # Если по текущей дате еще нет задач, инициализируем её
    if str(current_date) not in user_history['llm_queries']:
        user_history['llm_queries'][str(current_date)] = []

    today_tasks = user_history['llm_queries'][str(current_date)]

    # Получаем уникальный номер задачи
    last_task_number = len(today_tasks) + 1

    # Определяем ключ для новой задачи
    task_key = f'task_{last_task_number}'

    datetime_name = f"{current_date.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M%S')}"

    # Создаем новую запись для задания
    new_query_data = { 
        task_key: {
            "status": "Данные обрабатываются", 
            "total": 0, 
            "completed": 0,
            "percent": 0
        },
        'index': request.index,
        'filename': indexes[request.index] + '_' + datetime_name,
        'request_time': datetime_name,
        'system_prompt': request.system_prompt,
        'example_prompt': request.example_promt,
        'main_prompt': request.main_prompt,
        'min_date': request.min_date,
        'max_date': request.max_date,
        'query_str': request.query_str
    }

    # Добавляем новую задачу в текущую дату
    today_tasks.append(new_query_data)

    # Сохраняем обновленную историю
    save_history(request.user_id, user_history)

    # Добавление задачи в фон
    background_tasks.add_task(run_llm_query, request, task_key, indexes)
    
    return {"message": "Анализ запущен", "user_id": request.user_id, "task_key": task_key, "current_date": current_date}


async def run_llm_query(request: AnalysisRequest, task_key: str, indexes):

    await asyncio.sleep(0.01)
    et = time.time()
    # Получаем текущее время для добавления к имени файла
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    total_texts = 0  # Общее количество текстов

    if request.query_str != None:
        search = request.query_str.split(',')
        for i in range(len(search)):
            data = elastic_query(theme_index=indexes[request.index], query_str=search[i])

        # отфильтровываем по необходимой дате из календаря
        data = [x for x in data if request.min_date <= x['timeCreate'] <= request.max_date]

    else:
        data = elastic_query(theme_index=indexes[request.index], query_str='all')
        data = [x for x in data if request.min_date <= x['timeCreate'] <= request.max_date]

    ################################### data ###################################

    # Тексты для обработки
    texts = [x['text'] for x in data]
    texts = texts[:15]
    total_texts = len(texts)

    et = time.time()

    # Загрузка словаря истории запросов пользователей
    os.chdir('/home/dev/fastapi/analytics_app/data')
    with open('llm_history_progress.pickle', 'rb') as file:  # 'rb' - читать в бинарном формате
        search_history = pickle.load(file)

    print('Всего текстов: {}'.format(total_texts))

    ################################### model ###################################

    tokenizer = AutoTokenizer.from_pretrained("/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct")
    # model = AutoModelForCausalLM.from_pretrained("/home/dev/fastapi/analytics_app/data/LLM_models/Meta-Llama-3-8B-Instruct")

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

    ################################### promt ###################################
    if request.system_prompt == None:
        request.system_prompt = """
        <s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant for labeling topics.
        <</SYS>>
        """

    # Example prompt demonstrating the output we are looking for
    if request.example_promt == None:
        request.example_promt = """
        У меня есть следующий текст:
        ЕАЭС-Китай: обсуждены перспективы сотрудничества в таможенной сфере

        В штаб-квартире Евразийской экономической комиссии состоялась рабочая встреча министра по таможенному сотрудничеству ЕЭК Руслана Давыдова с таможенным советником Посольства Китайской Народной Республики в Российской Федерации Чжоу Вэньи.

        Стороны обсудили вопросы налаживания информационного обмена, развития цифровой таможни, внедрения современных технологий. Особое внимание уделено интеллектуальной таможне, применению навигационных пломб и механизма «единого окна». Отдельно обсуждены перспективы развития таможенной инфраструктуры с учетом лучших мировых практик, бесшовного транзита в рамках сотрудничества ЕАЭС и Китайской Народной Республики. Рассмотрены возможные подходы к выявлению причин и минимизации расхождений статистических данных в торговле.


        Тема описывается следующими ключевыми словами: 'министр, встреча, таможенное сотрудничество, ЕЭК, пломбы, ЕАЭС, Китай', цифровая таможня, интеллектуальная таможня.

        Основываясь на информации о ключевых словах выше, пожалуйста, випиши тематики этого текста. Убедитесь, что вы возвращаете только тематики и ничего больше.

        [/INST] Развитие транзита ЕАЭС, Встреча министра Руслана Давыдова, Внедрение современных технологий, интеллектуальная таможня
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

    # Параметр для отслеживания количества обработанных текстов
    completed_texts = 0

    # Определяем общее количество текстов для завершения
    total_texts = len(texts)
    llm_answer = []
    completed_texts = 0
    count = 0

    # Проверка схожести текстов: Для каждого текстового элемента проверяем, был ли он уже обработан. Если текст схож с ранее обработанными текстами (с использованием порога threshold), мы просто возвращаем кэшированный ответ.
    # Кэширование результатов: Для каждого проанализированного текста сохраняем результат в словаре processed_texts. Таким образом, если текст встречается повторно или если его похожесть превышает определенный порог, мы можем избежать повторного анализа.
    # Устранение повторных вычислений: Используем CountVectorizer и cosine_similarity, чтобы вычислить схожесть между новыми и существующими текстами, избегая необходимости в повторном запуске тяжелых вычислений, если текст уже был проанализирован.
    # Обновление прогресса: Прогресс обновляется после каждой итерации, чтобы показать статус обработки.
    
    # Словарь для хранения уже обработанных текстов и их тематик
    processed_texts = {}

    # Проверка схожести текстов
    def check_similarity_and_process(single_text, threshold=0.8):
        if single_text in processed_texts:
            return processed_texts[single_text]  # Возвращаем кэшированный ответ

        # Формируем DataFrame для анализа
        df_meta = pd.DataFrame({'text': [single_text]})
        
        # Проверка на существующие тексты
        if len(processed_texts) > 0:
            df_existing = pd.DataFrame(list(processed_texts.keys()), columns=['text'])
            combined_df = pd.concat([df_meta, df_existing], ignore_index=True)

            count_vectorizer = CountVectorizer()
            vector_matrix = count_vectorizer.fit_transform(combined_df['text'].values)
            cosine_similarity_matrix = cosine_similarity(vector_matrix)

            # Убираем диагональные элементы
            for i in range(len(cosine_similarity_matrix)):
                cosine_similarity_matrix[i][i] = 0

            # Проверяем на схожесть
            similar_indices = np.where(cosine_similarity_matrix[0] >= threshold)[0]
            if len(similar_indices) > 0:
                print('threshold сработал!')
                # Если есть похожие тексты, берем ответ от первого найденного
                return processed_texts[combined_df.iloc[similar_indices[0]]['text']]

        # Здесь выполняем анализ текста
        messages = [
            {
                "role": "system", 
                "content": request.system_prompt + request.example_promt + 
                'У меня есть следующий текст: ' + single_text + 
                ' Основываясь на информации о ключевых словах выше, пожалуйста, выпишите тематики этого текста. Убедитесь, что вы возвращаете только тематики и ничего больше. Отвечайте на русском языке.'
            }
        ]

        # Очищаем кэш перед вызовом модели
        torch.cuda.empty_cache()

        # Используем torch.no_grad() для предотвращения вычисления градиентов
        with torch.no_grad():
            response = pipe(messages, num_return_sequences=1)

        # Обрабатываем ответ
        result_text = response[0]['generated_text'][1]['content'].replace('[/INST]\n', '').replace('\n', '')

        # Кэшируем результат
        processed_texts[single_text] = result_text
        return result_text

    # Проходим через каждый текст по отдельности
    for i in range(total_texts):
        single_text = texts[i]

        if len(single_text) < 15000:
            answer = check_similarity_and_process(single_text)
            llm_answer.append(answer)
        else:
            llm_answer.append('Длинный текст')
            count += 1

        # Обновление прогресса
        completed_texts += 1
        percent = round((completed_texts / total_texts) * 100, 1)
        update_progress(request.user_id, task_key, {
            "status": "Данные обрабатываются", 
            "total": total_texts, 
            "completed": completed_texts,
            "percent": percent
        })

    # Завершение обработки
    update_progress(request.user_id, task_key, {
        "status": "Готово!", 
        "total": total_texts, 
        "completed": total_texts, 
        "percent": 100
    })

    # ################################### BERTopic ###################################

    # Pre-calculate embeddings
    embedding_model = SentenceTransformer("DeepPavlov/rubert-base-cased-sentence")
    embeddings = embedding_model.encode(llm_answer, show_progress_bar=True)


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
    # print("Лучшие параметры n_components:", n_components)
    # print("Лучшие параметры:", best_params)

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
    # keybert = KeyBERTInspired()

    # MMR
    # mmr = MaximalMarginalRelevance(diversity=0.3)


    if request.main_prompt == None:
        main_prompt = """
        [INST]
        У меня есть тема, содержащая следующие документы:
        [DOCUMENTS]

        Тема описывается следующими ключевыми словами: '[KEYWORDS]'.

        Основываясь на информации о теме выше, пожалуйста, создайте краткий заголовок этой темы. Убедитесь, что вы возвращаете только заголовок и ничего больше. Отвечай на русском языке.
        [/INST]
        """

    prompt = request.system_prompt + request.example_promt + request.main_prompt
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

    # Устанавливаем путь к директории файла
    file_location = f'/home/dev/fastapi/analytics_app/data/{request.user_id}/bertopic_files_directory/{request.folder_name}/'

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    # Сохранение файла .fig на диск
    # Формируем новое имя файла с добавлением даты и времени
    new_filename = f"{indexes[request.index]}_{current_time}.html"
    fig.write_html(file_location + new_filename)


    ###################################### save model #################################

    # Название для сохранения файлов
    filename = 'topic_model_' + new_filename.split('.html')[0]
    # print("!!!555!!!+++!!!555!!!")
    # print(filename)
    # print(new_filename.split('.html')[0])
    # print(new_filename)

    st = time.time()
    elapsed_time = st - et
    print(elapsed_time)
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

    # Получение и обработка данных пользователя
    user_data = redis_db.hgetall(request.user_id)

    if user_data:
        user_data_decoded = {key.decode('utf-8'): value.decode('utf-8') for key, value in user_data.items()}
        
        # Проверка на наличие 'bertopic_files_directory'
        if "bertopic_files_directory" in user_data_decoded:
            user_folders = json.loads(user_data_decoded["bertopic_files_directory"])
        else:
            user_folders = {}

        # Обработка и сохранение нового результата
        file_info = {
            "html-file": f"{indexes[request.index]}_{current_time}.html",
            "model-file": filename,
            "creation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "execution_time": execution_time 
        }

        folder_name = request.folder_name
        if folder_name not in user_folders:
            user_folders[folder_name] = []
        user_folders[folder_name].append(file_info)

        # Сохранение обратно в Redis
        redis_db.hset(request.user_id, "bertopic_files_directory", json.dumps(user_folders))
    else:
        raise Exception("User data does not exist.")

    return 'Анализ выполнен!'


# Эндпойнт для получения текущего прогресса llm-задачи
@app.get("/progress-llm/{user_id}/{date}/{task_id}")
async def get_progress(user_id: int, date: str, task_id: str):
    await asyncio.sleep(0.1)

    # Установка каталога для загрузки данных
    os.chdir('/home/dev/fastapi/analytics_app/data') 

    # Загрузка словаря истории запросов пользователей
    try:
        with open('llm_history_progress.pickle', 'rb') as file:
            search_history = pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл истории запросов не найден.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")

    # Поиск истории для указанного user_id
    user_history = next((item for item in search_history if item['user_id'] == user_id), None)

    if user_history is None:
        raise HTTPException(status_code=404, detail="История для данного пользователя не найдена.")

    # Извлечение запросов LLM для заданной даты
    llm_queries = user_history['values'].get('llm_queries', {}).get(date, None)

    if llm_queries is None:
        raise HTTPException(status_code=404, detail="Запросы для указанной даты не найдены.")

    # Поиск заданного task_id среди запросов
    task_progress = next((query for query in llm_queries if task_id in query), None)

    if task_progress is None:
        raise HTTPException(status_code=404, detail="Запрос с указанным task_id не найден.")

    # Извлечение статуса и прогресса задачи
    task_info = task_progress[task_id]

    # Формирование ответа
    response = {
        "task_id": task_id,
        "status": task_info['status'],
        "total_texts": task_info.get('total', 0),
        "completed_texts": task_info.get('completed', 0),
        "percent": task_info.get('percent', 0.0)
    }

    return response


# Функция для получения сессии базы данных
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session


from sqlalchemy.future import select

# JWT token scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get the current user based on the provided token
async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    print(payload)

    user_id: str = payload.get("sub") 
    return user_id
 

class ResponceModel(BaseModel):
    id:int
    class Config():
        orm_mode =True
    
# Route to retrieve the current user profile details
@app.get('/user-id', tags=['user'])
def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user

def get_user_profile(current_user: User = Depends(get_current_user)):
    return current_user


@app.get("/history_llm_search/{user_id}")
async def history_search(user_id: int):

    os.chdir('/home/dev/fastapi/analytics_app/data')
    
    # Загрузка словаря истории запросов пользователей
    try:
        with open('llm_history_progress.pickle', 'rb') as file:  # 'rb' - читать в бинарном формате
            search_history = pickle.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл истории запросов не найден.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке файла: {str(e)}")

    # Поиск по user_id
    user_requests = next((item for item in search_history if item['user_id'] == user_id), None)

    if user_requests:
        # Извлечение необходимой информации
        date = user_requests['values']['date']
        llm_queries = user_requests['values']['llm_queries']

        # Формирование ответа
        response = {
            "date": date,
            "llm_queries": llm_queries
        }
        return response
    else:
        raise HTTPException(status_code=404, detail="Запросы для данного пользователя не найдены.")


############################## Хранение данных о файлах и папках пользователей в Redis ####################

# Добавление папки
@app.get("/add-folder/{user_id}/{folder_name}")
async def add_folder(user_id: str, folder_name: str):
    # Путь до директории json_files
    json_files_directory = f"/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory"
    # Путь, где будет создана папка
    storage_path = f"{json_files_directory}/{folder_name}"

    # Проверяем, существует ли директория json_files_directory, если нет - создаем её
    if not os.path.exists(json_files_directory):
        os.makedirs(json_files_directory)

    # Проверяем, существует ли уже папка
    if os.path.exists(storage_path):
        raise HTTPException(status_code=400, detail="Папка с таким именем уже существует.")

    # Создаём папку
    os.makedirs(storage_path)

    # Получаем текущее состояние папок в Redis
    user_data = redis_db.hget(user_id, "json_files_directory")
    if user_data is None:
        user_folders = {}
    else:
        user_folders = json.loads(user_data)

    # Добавляем новую папку в структуру
    if folder_name not in user_folders:
        user_folders[folder_name] = []

    # Сохраняем обновлённую структуру в Redis
    redis_db.hset(user_id, "json_files_directory", json.dumps(user_folders))

    return f"Папка {folder_name} у пользователя {user_id} создана!"


# Добавление файла в папку
@app.post("/add-file/{user_id}/{folder_name}")
async def add_file(user_id: str, folder_name: str, uploaded_file: UploadFile = File(...)):
    # Проверка, что folder_name предоставлен
    if not folder_name:
        raise HTTPException(status_code=400, detail="Необходимо указать имя папки")
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    # Обновление словаря с темами
    # Новая строка для добавления
    new_value = uploaded_file.filename
    # Найдем следующий ключ
    next_key = max(indexes.keys()) + 1

    # Удаляем .json и переводим в нижний регистр
    formatted_value = new_value.replace('.json', '').lower()
    # Добавляем новое значение в словарь
    indexes[next_key] = formatted_value
    # Сохранение словаря с темами
    save_dict_to_pickle('/home/dev/fastapi/analytics_app/data/indexes.pkl', indexes)

    # Устанавливаем путь к директории файла
    file_location = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/{uploaded_file.filename.lower()}'
    
    # Проверка размера загружаемого файла
    max_file_size = 10 * 1024 * 1024 * 1024  # 10 GB
    if uploaded_file.size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail="Размер файла превышает допустимый предел 10 ГБ"
        )

    # Создание директории, если она не существует
    os.makedirs(os.path.dirname(file_location), exist_ok=True)

    # Проверка существования файла в папке
    user_folders_data = redis_db.hget(user_id, "json_files_directory")
    if user_folders_data is None:
        user_folders = {}
    else:
        user_folders = json.loads(user_folders_data)

    # Проверка существования файла в папке в Redis
    if uploaded_file.filename.lower() in user_folders[folder_name]:
        # Если файл существует, выводим сообщение о его существовании
        return f"Файл с именем '{uploaded_file.filename}' уже существует в папке '{folder_name}'."

        # Перезаписываем файл
        # Обновляем список файлов, добавляя файл заново
        # user_folders[folder_name].remove(uploaded_file.filename)  # Удаляем старую запись
        # user_folders[folder_name].append(uploaded_file.filename)   # Добавляем новый файл

        # return f"Файл с именем '{uploaded_file.filename}' успешно перезаписан в папке '{folder_name}'."

    # Сохранение файла на диск
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(uploaded_file.file, file_object)

    # Добавляем файл в список
    user_folders[folder_name].append(uploaded_file.filename.lower())
    
    # Сохраняем обновленный список в Redis
    redis_db.hset(user_id, "json_files_directory", json.dumps(user_folders))
    
    # Попытка загрузки файла в Elasticsearch
    # try:
    file_loc = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}/'
    load_file_to_elstic(uploaded_file, path=file_loc, next_key=str(next_key))
    # except Exception as e:
    #     raise HTTPException(status_code=400, detail="Загрузите, пожалуйста, валидный json из темы мониторинга")

    return f"Файл {uploaded_file.filename} загружен в папку {folder_name} пользователя - {user_id}!"


# Удаление папки
@app.delete("/delete-folder/{user_id}/{directory_type}/{folder_name}")
async def delete_folder(user_id: str, directory_type: str, folder_name: str):
    # Получаем текущее содержимое для указанного пользователя
    json_folders = redis_db.hget(user_id, directory_type)
    
    # Если данных для данного user_id нет, возвращаем ошибку
    if json_folders is None:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Декодируем JSON данные в словарь
    folders_dict = json.loads(json_folders)

    # Проверяем наличие запрашиваемой папки
    if folder_name not in folders_dict:
        raise HTTPException(status_code=404, detail="Запрашиваемая папка не найдена.")

    # Получаем список файлов, относящихся к этой папке
    files_in_directory = folders_dict[folder_name]

    # Удаляем папку из Redis
    del folders_dict[folder_name]  # Удаляем папку из словаря
    redis_db.hset(user_id, directory_type, json.dumps(folders_dict))  # Обновляем данные в Redis

    # Получаем список всех индексов для удаления из Elasticsearch
    es_indexes = [index for index in es.indices.get('*')]
    
    # Удаляем данные из Elasticsearch
    if files_in_directory:
        for file in files_in_directory:
            # Индекс, который нужно удалить
            index_to_delete = file.replace('.json', '')

            # Проверка существования индекса и его удаление
            if index_to_delete in es_indexes:
                es.indices.delete(index=index_to_delete)
                print(f"Индекс '{index_to_delete}' успешно удалён.")
            else:
                print(f"Индекс '{index_to_delete}' не найден.")

    # Формируем путь к удаляемой папке в файловой системе
    folder_path = f"/home/dev/fastapi/analytics_app/data/{user_id}/{directory_type}/{folder_name}"

    try:
        # Проверяем, существует ли папка
        if os.path.exists(folder_path):
            # Удаляем папку и всё её содержимое
            shutil.rmtree(folder_path)

            return {"message": f"Папка '{folder_name}' пользователя '{user_id}' успешно удалена."}
        else:
            raise HTTPException(status_code=404, detail="Папка не найдена.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Удаление файла
@app.delete("/delete-file/{user_id}/{directory_type}/{directory_name}/{file_name}")
async def delete_file(user_id: str, directory_type: str, directory_name: str, file_name: str):
    # Получаем директории для указанного user_id
    folders = redis_db.hgetall(user_id)
    # Преобразуем байтовые строки в обычные строки и десериализуем JSON
    folders = {key.decode('utf-8'): json.loads(value.decode('utf-8')) for key, value in folders.items()}

    # Проверяем, есть ли директории для данного пользователя
    if not folders:
        raise HTTPException(status_code=404, detail="Директории не найдены для данного пользователя.")

    # Определяем путь к директории файлов на диске
    folder_path = f"/home/dev/fastapi/analytics_app/data/{user_id}/{directory_type}/{directory_name}"

    # Удаляем файл из json_files_directory
    if directory_type == "json_files_directory":
        try:
            # Удаляем соответствующий словарь
            if directory_name in folders.get("json_files_directory", {}):
                schools_data = folders["json_files_directory"]
                # Ищем и удаляем словарь с необходимими файлами
                updated_schools = [item for item in schools_data[directory_name] if item != file_name + '.json']
                schools_data[directory_name] = updated_schools
                redis_db.hset(user_id, "json_files_directory", json.dumps(schools_data))

            # Удаляем файл из файловой системы
            os.remove(os.path.join(folder_path, file_name + '.json'))

            return {"message": f"Файл {file_name + '.json'} из директории {directory_name} был успешно удалён!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")


    # Удаляем файлы из bertopic_files_directory
    elif directory_type == "bertopic_files_directory":
        try:
            search_string = file_name.replace('topic_model_', '').replace('.html', '')
            # Удаляем соответствующий словарь
            print(folders.get("bertopic_files_directory", {}))
            if directory_name in folders.get("bertopic_files_directory", {}):
                schools_data = folders["bertopic_files_directory"]
                # Ищем и удаляем словарь с необходимым файлом
                updated_schools = [item for item in schools_data[directory_name] if item.get("html-file") != file_name]
                schools_data[directory_name] = updated_schools
                redis_db.hset(user_id, "bertopic_files_directory", json.dumps(schools_data))

            # Удаляем файлы
            file_pattern = os.path.join(folder_path, f"*{search_string}*")
            for f in glob.glob(file_pattern):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

            return {"message": f"Все файлы, содержащие {search_string}, из директории {directory_name} были успешно удалены!"}

        except Exception as e:
            print(f"Ошибка при удалении файлов: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")

    # Удаляем файлы из projector_files_directory
    elif directory_type == "projector_files_directory":
        try:
            search_string = file_name.replace('.txt', '').replace('.tsv', '')
            # Удаляем соответствующий словарь
            if directory_name in folders.get("projector_files_directory", {}):
                schools_data = folders["projector_files_directory"]
                # Ищем и удаляем словарь с необходимими файлами
                updated_schools = [
                    entry for entry in schools_data[directory_name]
                    if not (search_string in entry.get('tsv-file', '') or 
                            search_string in entry.get('txt-file', ''))
                ]
                schools_data[directory_name] = updated_schools
                redis_db.hset(user_id, "projector_files_directory", json.dumps(schools_data))

            # Удаляем файл и директорию с projector
            file_pattern = os.path.join(folder_path, f"*{search_string}*")
            for f in glob.glob(file_pattern):
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)

            return {"message": f"Все файлы, содержащие {search_string}, из директории {directory_name} были успешно удалены!"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка при удалении файлов: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="Некорректный тип директории.")


# # Переименование папки
# @app.put("/rename-folder/{user_id}/{old_folder_name}/{new_folder_name}")
# async def rename_folder(user_id: str, old_folder_name: str, new_folder_name: str):
#     # Путь до директории json_files
#     json_files_directory = f"/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory"
#     old_storage_path = f"{json_files_directory}/{old_folder_name}"
#     new_storage_path = f"{json_files_directory}/{new_folder_name}"

#     # Проверяем, существует ли старая папка
#     if not os.path.exists(old_storage_path):
#         raise HTTPException(status_code=404, detail="Старая папка не существует.")

#     # Проверяем, существует ли уже новая папка
#     if os.path.exists(new_storage_path):
#         raise HTTPException(status_code=400, detail="Папка с таким именем уже существует.")

#     # Переименовываем папку на файловой системе
#     os.rename(old_storage_path, new_storage_path)

#     # Обновляем информацию о папках в Redis
#     user_data = redis_db.hget(user_id, "json_folders")
#     if user_data is None:
#         raise HTTPException(status_code=404, detail="Данные пользователя не найдены.")

#     user_folders = json.loads(user_data)

#     # Переименовываем папку в структуре
#     if old_folder_name in user_folders:
#         user_folders[new_folder_name] = user_folders.pop(old_folder_name)
#     else:
#         raise HTTPException(status_code=404, detail="Старая папка не найдена в данных пользователя.")

#     # Сохраняем обновленную структуру в Redis
#     redis_db.hset(user_id, "json_folders", json.dumps(user_folders))

#     return f"Папка '{old_folder_name}' переименована в '{new_folder_name}' у пользователя {user_id}!"


# # Переименование файла
# @app.put("/rename-file/{user_id}/{folder_name}/{old_file_name}/{new_file_name}")
# async def rename_file(user_id: str, folder_name: str, old_file_name: str, new_file_name: str):
#     # Устанавливаем путь к директории файла
#     file_directory = f'/home/dev/fastapi/analytics_app/data/{user_id}/json_files_directory/{folder_name}'
#     old_file_path = f'{file_directory}/{old_file_name}'
#     new_file_path = f'{file_directory}/{new_file_name}'

#     # Проверяем, существует ли старая версия файла
#     if not os.path.exists(old_file_path):
#         raise HTTPException(status_code=404, detail="Старый файл не существует.")

#     # Проверяем, существует ли уже новая версия файла
#     if os.path.exists(new_file_path):
#         raise HTTPException(status_code=400, detail="Файл с таким именем уже существует в папке.")

#     # Переименовываем файл на файловой системе
#     os.rename(old_file_path, new_file_path)

#     # Обновляем информацию о файлах в Redis
#     user_folders_data = redis_db.hget(user_id, "json_folders")
#     if user_folders_data is None:
#         raise HTTPException(status_code=404, detail="Данные пользователя не найдены.")

#     user_folders = json.loads(user_folders_data)

#     # Проверка существования папки в Redis
#     if folder_name not in user_folders:
#         raise HTTPException(status_code=404, detail="Папка не найдена в данных пользователя.")

#     # Переименование файла в структуре
#     if old_file_name in user_folders[folder_name]:
#         user_folders[folder_name].remove(old_file_name)
#         user_folders[folder_name].append(new_file_name)
#     else:
#         raise HTTPException(status_code=404, detail="Старый файл не найден в папке.")

#     # Сохраняем обновленный список в Redis
#     redis_db.hset(user_id, "json_folders", json.dumps(user_folders))

#     return f"Файл '{old_file_name}' переименован в '{new_file_name}' в папке '{folder_name}' у пользователя {user_id}!"


# Эндпойнт получения папок и файлов для пользователя с данными из Elasticsearch
@app.get("/user-folders/{user_id}")
async def get_user_folders(user_id: str):
    # Проверяем, существует ли пользователь в БД
    user = get_user_profile(user_id)
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Путь к файлу с темами 
    file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
    # Загрузка словаря с темами
    indexes = load_dict_from_pickle(file_path)
    
    # Получаем папки пользователя из Redis
    folders = redis_db.hgetall(user_id)

    if not folders:
        return {"user_id": user_id, "json_files_directory": {}, "bertopic_files_directory": {}}
    
    # Преобразуем данные из Redis в формат JSON
    formatted_folders = {folder.decode('utf-8'): json.loads(files) for folder, files in folders.items()}

    # Получение данных из Elasticsearch
    es_indexes = [index for index in es.indices.get('*')] # список всех индексов elastic
    es_indexes = [x.strip() for x in es_indexes]

    # Запрос для поиска мин и макс дат в данных/файлах
    query = {
        "aggs": {
            "max_timeCreate": {
                "max": {
                    "field": "timeCreate"
                }
            },
            "min_timeCreate": {
                "min": {
                    "field": "timeCreate"
                }
            }
        },
    }

    # Создаем новый формат результата
    json_folders = {}

    # Проходим по всем именам папок в formatted_folders
    for folder_name in formatted_folders['json_files_directory'].keys():
        # Инициализируем ключ с пустым списком для каждой папки
        json_folders[folder_name] = []

    # Теперь обрабатываем файлы для каждой папки
    for folder_name, files in formatted_folders['json_files_directory'].items():
        for file_name in files:
            file_name = file_name.replace('.json', '').lower()

            # Проверяем, существует ли индекс для файла
            if file_name in es_indexes:
                date_period_query = es.search(index=file_name, body=query)['aggregations']

                json_folders[folder_name].append(
                    {
                        "file": file_name,
                        "min_data": date_period_query['min_timeCreate']['value'],
                        "max_data": date_period_query['max_timeCreate']['value'],
                        "index_number": list({i for i in indexes if indexes[i] == file_name})[0]
                    }
                )

    # Получаем папки пользователя из Redis для bertopic
    bertopic_folders = redis_db.hget(user_id, "bertopic_files_directory")
    
    # Если данные существуют и не пустые, обрабатываем их
    if bertopic_folders is not None:
        # Преобразуем данные из Redis в формат JSON
        try:
            # Поскольку redis_db.hget возвращает строку, нужно загрузить ее как JSON
            bertopic_folders = json.loads(bertopic_folders)
            # Преобразование в словарь, если требуется
            bertopic_folders = {folder: files for folder, files in bertopic_folders.items()}
        except json.JSONDecodeError:
            # Обработка случая, когда данные не валидные JSON
            bertopic_folders = {}

    else:
        bertopic_folders = {}

    # Получаем папки пользователя из Redis для bertopic
    projector_folders = redis_db.hget(user_id, "projector_files_directory")
    
    # Если данные существуют и не пустые, обрабатываем их
    if projector_folders is not None:
        # Преобразуем данные из Redis в формат JSON
        try:
            # Поскольку redis_db.hget возвращает строку, нужно загрузить ее как JSON
            projector_folders = json.loads(projector_folders)
            # Преобразование в словарь, если требуется
            projector_folders = {folder: files for folder, files in projector_folders.items()}
        except json.JSONDecodeError:
            # Обработка случая, когда данные не валидные JSON
            projector_folders = {}

    else:
        projector_folders = {}

    return {"user_id": user_id, "json_files_directory": json_folders, 
            "bertopic_files_directory": bertopic_folders, "projector_files_directory": projector_folders}
    

###########################################################################################################




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True) 
