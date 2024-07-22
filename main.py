import ast
from datetime import datetime
from enum import Enum
import re
import shutil
from typing import List, Optional, Union
from collections import ChainMap, defaultdict
import time
from os import listdir
from os.path import isfile, join
import tensorflow_hub as hub

import aiofiles
from sklearn import manifold
from fastapi_users import fastapi_users, FastAPIUsers
import pandas as pd
from pydantic import BaseModel, Field
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI, File, Request, UploadFile, status, Depends
from fastapi.encoders import jsonable_encoder
# from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import tensorflow_text

from typing import List, Union
from pydantic import BaseModel, Field
import functools as ft
import io

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
import codecs, json

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

es = Elasticsearch(
    ['localhost'],
    port=9200
)

path_json_files = '/home/dev/fastapi/analytics_app/data/json_files'

app = FastAPI(
    title="Analytics App"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load LLM
os.chdir('/home/dev/fastapi/analytics_app/data/LLM_models/')

model = "gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cpu",
) 
 

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
    text: str
    hub: str
    audienceCount: int
    commentsCount: int
    er: int
    url: str

# ModelAiAnalytics
class ModelAIAnalyticsNone(BaseModel):
    data: List[ModelItemAIAnalyticsNone]


class ModelAIPostAnalytics(BaseModel):
    id: int
    text: str
    llm_text: str


class ModelAIAnalyticsPost(BaseModel):
    promt: str
    texts: List[ModelAIPostAnalytics]


class QueryAiLLM(BaseModel):
    index: int=None
    min_date: int=None
    max_date: int=None
    promt: str = None
    texts_ids: list[int] = None

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

indexes = {1: "rosbank_01.02.2024-07.02.2024", 2: "skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024", 3:'rosbank_19.02.2024-29.02.2024', 
           4: "rosbank_14.03.2024-14.03.2024_fullday", 5: "r_13.03.2024-14.03.2024_full", 6: "rosbank_22.03.2024-24.03.2024", 
           7: "monitoring_tem_19.03.2024-25.03.2024", 8: 'rosbank_26.03.2024-01.04.2024', 9: 'tehfob', 10: 'transport_01.01.2024-09.04.2024', 
           11: 'moskovskiy_transport_01.01.2024_09.04.2024_2b', 12: 'rosbank_01.04.2024-15.04.2024', 13: 'rosbank_14.05.2024-16.05_чистая прибыль',
           14: 'contented_smi_01.04.2024-26.05.2024', 15: 'skillbox_smi_01.04.2024-26.05.2024', 16: 'rb_smi', 17: 'geekbrains', 18: 'eduson', 
           19: 'maley_nlmk_boevaya_tema_17.06.2024-21.06.2024_66757eb24cb15033866ecdd8', 20: 'maley_nlmk_boevaya_tema_17_06_2024_21_06_2024'}

 
@app.get('/data-users')
async def data_users(user: User = Depends(current_user)):

    es_indexes = [index for index in es.indices.get('*')] # список всех индексов elastic
    es_indexes = [x.strip() for x in es_indexes]

    # поиск мин и макс дат в данных/файлах
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

    if user.theme_rules == 'admin': # если пользователь админ, то вернуть все темы
        data_index = []
        for index in es_indexes:
            if index == 'read_me':
                continue
            date_period_query = es.search(index=index, body=query)['aggregations'] # запрос мин и макс дат в индексе
            try:
                data_index.append(
                    {
                        "file": index,
                        "min_data": date_period_query['min_timeCreate']['value'],
                        "max_data": date_period_query['max_timeCreate']['value'],
                        "index_number": list({i for i in indexes if indexes[i]==index})[0]
                    }
                )
            except:
                continue

        data_index = sorted(data_index, key=lambda d: d['index_number'])       
        return {"values": data_index}
    
    else: # если пользователь не админ, то вернуть его темы
        data_index = []
        user_index = list(set(es_indexes) & set([x.strip().lower().replace('.json', '') for x in user.theme_rules.split(',')]))

        for index in user_index:
            if index == 'read_me':
                continue
            date_period_query = es.search(index=index, body=query)['aggregations'] # запрос мин и макс дат в индексе
            data_index.append(
                {
                    "file": index,
                    "min_data": date_period_query['min_timeCreate']['value'],
                    "max_data": date_period_query['max_timeCreate']['value'],
                    "index_number": list({i for i in indexes if indexes[i]==index})[0]
                }
            )
        
        data_index = sorted(data_index, key=lambda d: d['index_number'])
        return {"values": data_index}


@app.post("/uploadfile/") # метод загрузки файлов json
async def create_upload_file(file: UploadFile = File(...)):

    file_location = path_json_files + "/" + file.filename
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    load_file_to_elstic(file, path=path_json_files)
    return {"filename": file.filename}


@app.get("/tonality_landscape")
async def tonality_landscape(user: User = Depends(current_user), index: int =None, 
                             min_date: int=None, max_date: int=None) -> Model_TonalityLandscape:
    
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
    
    os.chdir('/home/dev/fastapi/analytics_app/files')
    # данные с описанием тематик
    filename = indexes[index] + '_LLM'
    with open (filename, 'rb') as fp:
        data = pickle.load(fp)

    print(data[:2])

    data = [x[0]['generated_text'].split('model\n')[1] if len(x) == 1 else x for x in data]
    data = pd.DataFrame(data[:2])

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

    print(data)

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

    print('ThemesModel')
  
    return ThemesModel(values=fin_data)


@app.get("/voice")
async def voice_analize(user: User = Depends(current_user), index: int = None, 
                             min_date: int=None, max_date: int=None, query_str: str = None) -> ModelVoice:
    
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
        dff['timeCreate'] = [datetime.datetime.fromtimestamp(x).strftime(
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

    # data = {"first_graph":{
    #     "neg_smi_name": list_neg_smi,
    #     "neg_smi_count": list_pos_smi_massage_count,
    #     "neg_smi_rating": list_neg_smi_index,
    #     "pos_smi_name": list_pos_smi,
    #     "pos_smi_count": list_pos_smi_massage_count,
    #     "pos_smi_rating": list_pos_smi_index},

    #     "second_graph":{ "date_bobble": [x[0] for x in bobble],
    #     "name_bobble": [x[1] for x in bobble],
    #     "index_bobble": [x[2] for x in bobble],
    #     "z_index_bobble": [1] * len(bobble),
    #     "tonality_index_bobble": [x[3] for x in bobble],
    #     "tonality_url": [x[4] for x in bobble]}
    # }

    values = {}
    values['first_graph'] = {}
    values['first_graph']['negative_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_neg_smi, list_neg_smi_index, list_neg_smi_massage_count)]
    values['first_graph']['positive_smi'] = [{'name': x, "index": y, "message_count": z} for (x, y, z) in zip(list_pos_smi, list_pos_smi_index, list_pos_smi_massage_count)]

    values['second_graph'] = ''
    values['second_graph'] = [{'name': x, 'time': y, 'index': z, 'url': u,'color': t} for (x,y,z,u,t) in zip([x[1] for x in bobble], [x[0] for x in bobble], [x[2] for x in bobble], [x[4] for x in bobble], [x[3] for x in bobble])]

    return MediaRatingModel(first_graph=values['first_graph'], second_graph=values['second_graph'])


@app.get('/ai-analytics')
async def ai_analytics_get(index: int=None, min_date: int=None, max_date: int=None) -> ModelAIAnalyticsNone:
    
    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[index], query_str='all')

    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if min_date <= x['timeCreate'] <= max_date]
    keys = ['id', 'text', 'hub', 'audienceCount', 'commentsCount', 'er', 'url'] # ключи для отображения в первой таблице
    data = [{k: y.get(k, None) for k in keys} for y in data[:100]] # данные для первой таблицы
    ranges = list(np.arange(0, len(data)))
    [x.update({'id': y.item()}) for x, y in zip(data, ranges)] # меняем значение id на 0,1,2...для передачи далее при выборе на LLM

    return ModelAIAnalyticsNone(data=data)


@app.post('/ai-analytics')
async def ai_analytics_post(query: QueryAiLLM) -> ModelAIAnalyticsPost:

    st = time.time()

    # делаем запрос на текстовый поиск
    data = elastic_query(theme_index=indexes[query.index], query_str='all')
    # отфильтровываем по необходимой дате из календаря
    data = [x for x in data if query.min_date <= x['timeCreate'] <= query.max_date]
    
    # если был введен промт и выбраны строчки текстов (индексты в таблице данных), то начать запрос к LLM
    # query.promt = "Какая тематика у этого текста? Текст: "
    data = [data[x] for x in query.texts_ids]

    LLM = {} # данные для возврата вида {'text': 'описание'}
    LLM['promt'] = query.promt
    LLM['texts'] = []

    # print(query.promt)
    # print(data)
    
    for i in range(len(data)): # цикл работы LLM с выбранными текстами
        messages = [
            {"role": "user", "content": query.promt + data[i]['text']},
        ]
        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipeline(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            batch_size=128 
        )
        LLM['texts'].append({'id': query.texts_ids[i], 'text': data[i]['text'], 
                                'llm_text': outputs[0]['generated_text'].split('model\n')[1]})

    print('Done!')
    print(LLM['texts'])

    # get the end time
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    # ans = {
    #     "promt": "Какая тематика у этого текста? Текст: ",
    #     "texts": [
    #         {
    #         "id": 1,
    #         "text": "Комментирует директор территориального офиса Росбанка в Ярославле Татьяна Панова.\nЕсли вы планируете взять кредит в банке, предварительно оцените текущую нагрузку на бюджет. Сравните кредитные предложения разных банков (программу кредитования, удобство погашения, качество сервиса), чтобы выбрать подходящие именно для вас условия.\nВнимательно читайте кредитный договор и приложения к нему. Он содержит общие условия, которые устанавливаются банком в одностороннем порядке, и индивидуальные - они должны быть согласованы с заемщиком и содержать информацию обо всех обязательствах сторон.\nПроверьте эффективную процентную ставку и полную стоимость кредита (ПСК). Она вычисляется в годовых процентах и учитывает платежи заемщика по кредитному договору, размеры и сроки уплаты которых известны на момент его заключения. В полную стоимость кредитов включаются: сумма основного долга, комиссии, проценты, платежи в пользу третьих лиц (если предусмотрены договором), платежи по страхованию (если, например, от них зависит процентная ставка). В ПСК не включаются платежи, обусловленные законом, зависящие от решений заемщика, по страхованию залога, штрафы, пени.        \nОбратите внимание на график платежей. Например, кредит с аннуитетными платежами имеет меньшую ежемесячную нагрузку на бюджет, нежели кредит с дифференцированными, но общая сумма переплаты за весь срок кредита будет выше. Ознакомьтесь с суммой дополнительных сборов и комиссий, например, за снятие наличных в банкоматах, смс- уведомления, предоставление выписки по счету и другие операции.\nЖелательно узнать условия досрочного погашения: наличие комиссии, пересчитывается ли срок кредита или сумма платежа. Перерасчет срока кредита намного выгоднее.\nПодойдите осознанно к принятию решения о кредитной нагрузке. Кредит – это долг, который необходимо будет вернуть, поэтому оцените все плюсы и минусы.",
    #         "llm_text": "Тема текста – анализ кредитных offering bank, сравнительный анализ кредитных продуктов разных банков."
    #         },
    #         {
    #         "id": 3,
    #         "text": "Банки продолжают придумывать всё новые акции по своим продуктам, что бы как можно больше выдать кредитных карт новым пользователям. И февраль 2024 года не исключение. Мы рассмотрели самые интересные из них. Фото: BankiClub.ru Список акций по кредитным картам за февраль 2024 года Росбанк Мир «#120наВСЁ» Плюс - кэшбэк 3000 ₽ за покупки на Маркетплейсах  Акция: Оформите кредитную карту Мир #120наВСЁ Плюс до 29 февраля. Оплатите покупки картой на Ozon, Wildberries и «Яндекс Маркет» на общую сумму от 9000 ₽. Получите кешбэк 3 000 ₽ который поступит на карту до 31 марта.  Оформить карту Росбанк Мир «#120наВСЁ» Плюс Tinkoff «Platinum» - кэшбек 2000 рублей  Акция: Оформите карту и банк вернёт 2000 ₽, если потратите 5000 ₽ за месяц после активации карты.  Оформить карту Tinkoff «Platinum» ВТБ «Карта возможностей» - кешбэк 20% на все покупки  Акция: ВТБ дарит кэшбэк в размере 20% за покупки, совершенные в первые 30 дней обслуживания. Обратите внимание на то, что максимальная сумма вознаграждения составляет 2 тысячи бонусных рублей. Оформить карту ВТБ «Карта возможностей» МТС «Cashback» - до 13 месяцев без %  Акция: До 13 месяцев без % на сумму первой покупки, совершенной в течение 30 дней с даты выдачи карты в магазинах МТС Оформить карту МТС «Cashback» Свой Банк «Своя кредитка» - 2000 рублей в подарок в Магните  Акция: Совершите по карте покупку на любую сумму до 31.03.2024 и получите подарочный сертификат на сумму 2000 ₽ в Магнит. Оформить карту Свой Банк «Своя кредитка» Tinkoff «Drive» - бесплатное годовое обслуживание  Акция: Оформите кредитную карту Tinkoff Drive до 29 февраля и получите вечное бесплатное годовое обслуживание. Оформить карту Tinkoff «Drive» Хоум банк «120 дней без %» - 1000 баллов для новых клиентов  Акция: 1000 баллов Польза за покупки от 1000 рублей для новых клиентов (1 балл = 1 рубль);\nБесплатное обслуживание;\nБесплатные переводы и снятие наличных в течение 30 дней;\nОформить карту Хоум банк «120 дней без %» Tinkoff «ALL Airlines» - кэшбэк 10% на категорию ж/д билетов!  Акция: Банк начислит 10% бонусами за покупки в категории ж/д билетов (MCC:4011, 4112, 4722) в течение 30 дней с даты активации карты (в период активации с 01.02.2024 по 29.02.2024) по промокоду \"VAGON24\". Максимум - 1000 бонусов. Бонус предоставляется, если у вас нет или в течение года не было других кредитных карт Банка. Оформить карту Tinkoff «ALL Airlines» Европа Банк «URBAN CARD» - 1000 рублей за онлайн-заявку и 5% кэшбэк  Акция: Банк дарит 1 тысячу приветственных баллов при заполнении заявки через Госуслуги. Оформить карту Европа Банк «URBAN CARD» МТС Деньги «Zero» - 0% годовых  Акция: Вы платите всего 59 ₽ в день, только если пользуетесь кредитом. Оформить карту МТС Деньги «Zero» Европа Банк «Metro» - 1000 рублей за онлайн-заявку  Акция: Банк дарит 1 тысячу приветственных баллов при заполнении заявки через Госуслуги. Оформи карту через Госуслуги;\nпотрать 10 000 руб. на покупки;\nполучи 1000 руб. в подарок.\nОформить карту Европа Банк «Metro» Tinkoff «ALL Games» - кэшбэк 50% за покупки фастфуда  Акция: Тинькофф Банк дарит кэшбэк 50% по промокоду \"HUNGERKOMBAT24\" на покупки в категории фастфуда (MCC:5814) в течение 30 дней с даты активации карты. Максимум - 1000 бонусов. Оформить карту Tinkoff «ALL Games» Подписывайтесь на наш канал в Телеграме: актуальные акции, новости из мира финансов, конкурсы!  Источник: https://bankiclub.ru/bankovskie-karty/kreditnye-karty/kreditnye-karty-s-samymi-vygodnymi-aktsiyami-2022-goda/",
    #         "llm_text": "Тема текста - Акции по кредитным картам в Росбаке и ВТБ."
    #         }
    #     ]
    #     }
    # return ans
    return ModelAIAnalyticsPost(promt=LLM['promt'], texts=LLM['texts'])

### Model Competitors
class QueryCompetitors(BaseModel):
    themes_ind: list[int]=None
    min_date: int=None
    max_date: int=None


class FirstGraphValue(BaseModel):
    timestamp: int
    count: int


class FirstGraphCompetitors(BaseModel):
    index_name: str
    values: List[FirstGraphValue]


class SocmediaCompetitors(BaseModel):
    name: str
    count: int
    rating: int


class SecondGraphCompetitors(BaseModel):
    index_name: str
    SMI: List
    Socmedia: List[SocmediaCompetitors]


class SocmediaThirdGraphCompetitors(BaseModel):
    date: int
    name: str
    rating: int
    url: str


class ThirdGraphCompetitors(BaseModel):
    index_name: str
    SMI: List
    Socmedia: List[SocmediaThirdGraphCompetitors]


class CompetitorsModel(BaseModel):
    first_graph: List[FirstGraphCompetitors]
    second_graph: List[SecondGraphCompetitors]
    third_graph: List[ThirdGraphCompetitors]



@app.post('/competitors')
async def competitors(query: QueryCompetitors) -> CompetitorsModel:

    another_graph = []

    min_date = []
    max_date = []
    themes_ind = query.themes_ind

    for i in range(len(themes_ind)):

        # делаем запрос на текстовый поиск
        data = elastic_query(theme_index=indexes[themes_ind[i]], query_str='all')
        # отфильтровываем по необходимой дате из календаря
        ind_df = [x for x in data if query.min_date <= x['timeCreate'] <= query.max_date]
        # заменяем audience на audienceCount для случаев только СМИ
        ind_df = [
            {"audienceCount" if k == "audience" else k: v for k, v in x.items()} for x in ind_df]

        # добавляем в СМИ объект 'authorObject'
        for i in range(len(ind_df)):
            if 'authorObject' not in ind_df[i]:
                authorObject = {}
                authorObject['url'] = ind_df[i]['url']
                authorObject['author_type'] = ind_df[i]['categoryName']
                authorObject['sex'] = 'n/a'
                authorObject['age'] = 'n/a'
                ind_df[i]['authorObject'] = authorObject

            if 'hubtype' not in ind_df[i]:
                ind_df[i]['hubtype'] = 'Онлайн-СМИ'

        # поиск минимальной и максимальной даты (самой ранней и самой поздней)
        min_date.append(
            np.min([x['timeCreate'] for x in [item for item in ind_df]]))
        max_date.append(
            np.max([x['timeCreate'] for x in [item for item in ind_df]]))
        another_graph.append(ind_df)

    dates = [min_date, max_date]
    min_date = np.min(dates[0])
    max_date = np.max(dates[1])

    filenames = [indexes[x] for x in themes_ind].copy()
    name_files = [indexes[x] for x in themes_ind]

    # для фиксации далее файла, в котором будет минимальная/начальная дата для старта графика
    min_date_number_file = dates[0].index(min_date)

    for i in range(len(another_graph)):

        themes_ind[i] = pd.DataFrame(another_graph[i])
        # метаданные
        themes_ind[i] = themes_ind[i][['text', 'url',
                             'hub', 'timeCreate', 'audienceCount']]

        #     themes_ind[i] = themes_ind[i].set_index(['timeCreate'])  # индекс - время создания поста

        themes_ind[i]['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                  themes_ind[i]['timeCreate'].values]

        themes_ind[i] = themes_ind[i].set_index(['timeCreate'])

        themes_ind[i]['timeCreate'] = list(themes_ind[i].index)
        themes_ind[i]['timeCreate'] = pd.to_datetime(themes_ind[i]['timeCreate'])

        # разбиваем даные на 10-минутки (отрезки по 10 мин для группировки далее)
        bins = pd.date_range(datetime.fromtimestamp(min_date).strftime('%Y-%m-%d %H:%M:%S'),
                             datetime.fromtimestamp(max_date).strftime('%Y-%m-%d %H:%M:%S'), freq='10T')
        themes_ind[i]['bins'] = pd.cut(themes_ind[i]['timeCreate'], bins)


        themes_ind[i]['bins_left'] = pd.IntervalIndex(
            pd.cut(themes_ind[i]['timeCreate'], bins)).left
        themes_ind[i]['bins_right'] = pd.IntervalIndex(
            pd.cut(themes_ind[i]['timeCreate'], bins)).right

        themes_ind[i]['bins_time_unix'] = themes_ind[i]['bins_left'].to_numpy().astype(
            np.int64) // 10 ** 9

        themes_ind[i].index = themes_ind[i]

        # оставляем только время
        themes_ind[i] = themes_ind[i][['bins_time_unix', 'bins_left', 'bins_right']]

        # подсчет кол-ва собщений за каждые n минут
        themes_ind[i] = pd.DataFrame(
            themes_ind[i][['bins_time_unix']].value_counts())
        themes_ind[i]['bins_time_unix'] = [x[0]
                                      for x in list(themes_ind[i].index)]
        themes_ind[i].columns = ['count', 'time']
        themes_ind[i] = themes_ind[i][['time', 'count']]
        # сортируем по дате от минимальной до конечной, убираем 1ую строчку (с отрицат.значением времени)
        themes_ind[i] = themes_ind[i].sort_values(by='time').loc[1:]
        themes_ind[i].columns = ['time_' + str(i), filenames[i]]

    # объединяем данные в финальную таблицу
    # добавляем колонку с индексом для объединения талиц в 1 табл.
    for i in range(len(themes_ind)):
        themes_ind[i]['index_col'] = [x[0] for x in themes_ind[i].index]

    # ф-ия объединения нескольких таблиц в одну
    df_fin = ft.reduce(lambda left, right: pd.merge(
        left, right, on=['index_col']), themes_ind)

    # убираем одинаковые столбцы с датами и index_col (такие же значения есть в индексе), оставляем кол-во сообщений за каждые 10мин
    df_fin = df_fin[[x for x in df_fin.columns if 'time' not in x]]
    df_fin.index = df_fin['index_col']
    df_fin = df_fin[[
        x for x in df_fin.columns if 'index_col' not in x]]

    df_fin['date'] = [x for x in df_fin.index]
    df_fin = df_fin[df_fin.columns[::-1]]

    # приводим дату к unix-формату
    a = [datetime.fromtimestamp(x).strftime(
        '%Y-%m-%d %H:%M:%S') for x in df_fin['date'].values]
    a = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') -
              datetime(1970, 1, 1)).total_seconds() * 1000) for x in a]
    df_fin['date'] = a

    data_file = [y.replace('.json', '') for y in df_fin.columns if y != 'date'] # имена файлов с данными
    first_graph = []
    for i in range(len(data_file)):
        data = {}
        data['index_name'] = data_file[i]
        data['values'] = [dict(zip(df_fin['date'], df_fin[x])) for x in [y for y in df_fin.columns if y != 'date']]
        first_graph.append(data)

    for i in range(len(first_graph)): # приводим данные к виду: {'Monitoring_tem_11.05.2024-13.05.2024_y': [{'timestamp': 1715397081000, 'count': 1}, {'timestamp': 1715397681000, 'count': 1}
        first_graph[i]['values'] = [{'timestamp': x, 'count': y} for x, y in zip(first_graph[i]['values'][0].keys(), 
                                                                                    first_graph[i]['values'][0].values())]
    

    #### second graph - график получения шариков с размерами по индексам (аудитория иили индекс СМИ)
    # bobbles
    a = None
    datarange = False
    second_graph_smi = []
    second_graph_socmedia = []

    data = {}
    data["news_smi_name"] = []
    data["news_smi_count"] = []
    data["news_smi_rating"] = []
    count_data = 50  # сколько данных/кругов выводить внутри

    for i in range(len(another_graph)):

        themes_ind[i] = pd.DataFrame(another_graph[i])
        columns = ['citeIndex', 'timeCreate', 'toneMark',
                   'hubtype', 'hub', 'audienceCount', 'url']

        # метаданные
        themes_ind[i] = pd.concat([pd.DataFrame.from_records(themes_ind[i]['authorObject'].values), themes_ind[i][columns]],
                             axis=1)

        themes_ind[i]['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                  themes_ind[i]['timeCreate'].values]

        # индекс - время создания поста
        themes_ind[i] = themes_ind[i].set_index(['timeCreate'])

        if set(themes_ind[i]['hub'].values) == {"telegram.org"}:

            a = themes_ind[i]
            a = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['hub'] == "telegram.org")]

            # negative smi
            df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)][
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
            list_neg = [int(x[0]) for x in list_neg]

            for i in range(len(list_neg)):
                dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

            dict_neg = dict(
                sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

            dict_neg_hubs_count = dict(
                Counter(list(
                    a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)]['fullname'])))

            fin_neg_dict = defaultdict(tuple)
            # you can list as many input dicts as you want here
            for d in (dict_neg, dict_neg_hubs_count):
                for key, value in d.items():
                    fin_neg_dict[key] += (value,)

            list_neg_smi = list(fin_neg_dict.keys())
            list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
            list_neg_smi_massage_count = [x[1]
                                          for x in fin_neg_dict.values()]

            # positive smi
            df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)][
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
            list_pos = [int(x[0]) for x in list_pos]

            for i in range(len(list_pos)):
                dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

            dict_pos = dict(
                sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

            dict_pos_hubs_count = dict(
                Counter(list(
                    a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)]['fullname'])))

            fin_pos_dict = defaultdict(tuple)
            # you can list as many input dicts as you want here
            for d in (dict_pos, dict_pos_hubs_count):
                for key, value in d.items():
                    fin_pos_dict[key] += (value,)

            list_pos_smi = list(fin_pos_dict.keys())
            list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
            list_pos_smi_massage_count = [x[1]
                                          for x in fin_pos_dict.values()]

            # data to bobble graph
            bobble = []
            df_tonality = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                ['fullname', 'audienceCount', 'toneMark', 'url']].values
            index_ton = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append(
                        [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                elif df_tonality[i][2] == 1:
                    bobble.append(
                        [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

            for i in range(len(bobble)):
                if bobble[i][3] == 1:
                    bobble[i][3] = "#32ff32"
                else:
                    bobble[i][3] = "#FF3232"

            # list_neg_smi = [words_only(x) for x in list_neg_smi]
            # list_pos_smi = [words_only(x) for x in list_pos_smi]
            name_bobble = [x[1] for x in bobble]
            # name_bobble = [words_only(x) for x in name_bobble]

            data = {
                "neg_smi_name": list_neg_smi[:100],
                "neg_smi_count": list_pos_smi_massage_count[:100],
                "neg_smi_rating": list_neg_smi_index[:100],
                "pos_smi_name": list_pos_smi[:100],
                "pos_smi_count": list_pos_smi_massage_count[:100],
                "pos_smi_rating": list_pos_smi_index[:100],

                "date_bobble": [x[0] for x in bobble],
                "name_bobble": name_bobble,
                "index_bobble": [x[2] for x in bobble],
                "z_index_bobble": [1] * len(bobble),
                "tonality_index_bobble": [x[3] for x in bobble],
                "tonality_url": [x[4] for x in bobble],
            }

            data = json.dumps(data)
            filenames = ', '.join(filenames)

    #         return render_template('competitors.html', len_themes_ind=len_themes_ind, themes_ind=json_themes_ind, data=data, filenames=filenames, date=date)

        a = themes_ind[i]
        themes_ind[i] = themes_ind[i][themes_ind[i]['hubtype'] == 'Онлайн СМИ']

        # smi
        df_hub_siteIndex = a[a['hubtype'] ==
                             'Онлайн-СМИ'][['hub', 'citeIndex']].values

        dict_news = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_news.keys():

                dict_news[df_hub_siteIndex[i][0]] = []
                dict_news[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

            else:
                dict_news[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

        dict_news_hubs_count = dict(
            Counter(list(a[a['hubtype'] == 'Онлайн-СМИ']['hub'])))

        fin_news_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_news, dict_news_hubs_count):
            for key, value in d.items():
                fin_news_dict[key] += (value,)

        list_news_smi = list(fin_news_dict.keys())
        list_news_smi_index = [x[0] for x in fin_news_dict.values()]
        list_news_smi_massage_count = [x[1]
                                       for x in fin_news_dict.values()]

        # убираем n/a, берем максимальное значение индекса СМИ за период
        for i in range(len(list_news_smi_index)):
            list_news_smi_index[i] = [
                int(x) for x in list_news_smi_index[i] if type(x) != str]

            if list_news_smi_index[i] == []:
                list_news_smi_index[i] = [0]
            list_news_smi_index[i] = np.max(list_news_smi_index[i])

        data["news_smi_name"].append(list_news_smi[:count_data])
        data["news_smi_count"].append(
            list_news_smi_massage_count[:count_data])
        data["news_smi_rating"].append(
            [int(x) for x in list_news_smi_index][:count_data])

        datas = {
            "news_smi_name": data["news_smi_name"],
            "news_smi_count": data["news_smi_count"],
            "news_smi_rating": data["news_smi_rating"]
        }
        
        second_graph_smi.append([{'name': x, 'count': y, 'rating': z} for x,y,z in zip(datas['news_smi_name'][0], 
                                                        datas['news_smi_count'][0], datas['news_smi_rating'][0])])
        

    # bubble2 socmedia
    a = None
    data = {}
    data["list_socmedia"] = []
    data["list_socmedia_massage_count"] = []
    data["list_socmedia_index"] = []

    for i in range(len(another_graph)):

        themes_ind[i] = pd.DataFrame(another_graph[i])
        columns = ['citeIndex', 'timeCreate', 'toneMark',
                   'hubtype', 'hub', 'audienceCount', 'url']

        # метаданные
        themes_ind[i] = pd.concat([pd.DataFrame.from_records(themes_ind[i]['authorObject'].values), themes_ind[i][columns]],
                             axis=1)

        themes_ind[i]['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                  themes_ind[i]['timeCreate'].values]

        # индекс - время создания поста
        themes_ind[i] = themes_ind[i].set_index(['timeCreate'])

        if set(themes_ind[i]['hub'].values) == {"telegram.org"}:

            a = themes_ind[i]
            a = a[(a['hubtype'] == 'Мессенджеры каналы')
                  & (a['hub'] == "telegram.org")]

            # negative smi
            df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)][
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
            list_neg = [int(x[0]) for x in list_neg]

            for i in range(len(list_neg)):
                dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

            dict_neg = dict(
                sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

            dict_neg_hubs_count = dict(
                Counter(list(
                    a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == -1)]['fullname'])))

            fin_neg_dict = defaultdict(tuple)
            # you can list as many input dicts as you want here
            for d in (dict_neg, dict_neg_hubs_count):
                for key, value in d.items():
                    fin_neg_dict[key] += (value,)

            list_neg_smi = list(fin_neg_dict.keys())
            list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
            list_neg_smi_massage_count = [x[1]
                                          for x in fin_neg_dict.values()]

            # positive smi
            df_hub_siteIndex = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)][
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
            list_pos = [int(x[0]) for x in list_pos]

            for i in range(len(list_pos)):
                dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

            dict_pos = dict(
                sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

            dict_pos_hubs_count = dict(
                Counter(list(
                    a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] == 1)]['fullname'])))

            fin_pos_dict = defaultdict(tuple)
            # you can list as many input dicts as you want here
            for d in (dict_pos, dict_pos_hubs_count):
                for key, value in d.items():
                    fin_pos_dict[key] += (value,)

            list_pos_smi = list(fin_pos_dict.keys())
            list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
            list_pos_smi_massage_count = [x[1]
                                          for x in fin_pos_dict.values()]

            # data to bobble graph
            bobble = []
            df_tonality = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                ['fullname', 'audienceCount', 'toneMark', 'url']].values
            index_ton = a[(a['hubtype'] == 'Мессенджеры каналы') & (a['toneMark'] != 0)][
                ['timeCreate']].values.tolist()
            date_ton = [x[0] for x in index_ton]
            date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1,
                                                                                                    1)).total_seconds() * 1000)
                        for x in date_ton]

            for i in range(len(df_tonality)):
                if df_tonality[i][2] == -1:
                    bobble.append(
                        [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
                elif df_tonality[i][2] == 1:
                    bobble.append(
                        [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

            for i in range(len(bobble)):
                if bobble[i][3] == 1:
                    bobble[i][3] = "#32ff32"
                else:
                    bobble[i][3] = "#FF3232"

            # list_neg_smi = [words_only(x) for x in list_neg_smi]
            # list_pos_smi = [words_only(x) for x in list_pos_smi]
            name_bobble = [x[1] for x in bobble]
            # name_bobble = [words_only(x) for x in name_bobble]

            data = {
                "neg_smi_name": list_neg_smi[:100],
                "neg_smi_count": list_pos_smi_massage_count[:100],
                "neg_smi_rating": list_neg_smi_index[:100],
                "pos_smi_name": list_pos_smi[:100],
                "pos_smi_count": list_pos_smi_massage_count[:100],
                "pos_smi_rating": list_pos_smi_index[:100],

                "date_bobble": [x[0] for x in bobble],
                "name_bobble": name_bobble,
                "index_bobble": [x[2] for x in bobble],
                "z_index_bobble": [1] * len(bobble),
                "tonality_index_bobble": [x[3] for x in bobble],
                "tonality_url": [x[4] for x in bobble],
            }

            data = json.dumps(data)

            filenames = ', '.join(filenames)

    #         return render_template('competitors.html', len_themes_ind=len_themes_ind, themes_ind=json_themes_ind, data=data, filenames=filenames, date=date)

        a = themes_ind[i]
        themes_ind[i] = themes_ind[i][themes_ind[i]['hubtype'] != 'Онлайн-СМИ']

        df_hub_siteIndex = a[a['hubtype'] != 'Онлайн-СМИ'][[
            'hub', 'audienceCount']].values

        dict_socmedia = {}
        for i in range(len(df_hub_siteIndex)):

            if df_hub_siteIndex[i][0] not in dict_socmedia.keys():

                dict_socmedia[df_hub_siteIndex[i][0]] = []
                dict_socmedia[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

            else:
                dict_socmedia[df_hub_siteIndex[i][0]].append(
                    df_hub_siteIndex[i][1])

        dict_socmedia_hubs_count = dict(
            Counter(list(a[a['hubtype'] != 'Онлайн-СМИ']['hub'])))

        fin_news_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_socmedia, dict_socmedia_hubs_count):
            for key, value in d.items():
                fin_news_dict[key] += (value,)

        list_socmedia = list(fin_news_dict.keys())
        list_socmedia_index = [x[0] for x in fin_news_dict.values()]
        list_socmedia_massage_count = [x[1]
                                       for x in fin_news_dict.values()]

        # убираем n/a, берем максимальное значение индекса СМИ за период
        for i in range(len(list_socmedia_index)):
            list_socmedia_index[i] = [
                int(x) for x in list_socmedia_index[i] if type(x) != str]

            if list_socmedia_index[i] == []:
                list_socmedia_index[i] = [0]
            list_socmedia_index[i] = np.max(list_socmedia_index[i])

        data["list_socmedia"].append(list_socmedia[:count_data])
        data["list_socmedia_massage_count"].append(
            list_socmedia_massage_count[:count_data])
        data["list_socmedia_index"].append(
            list_socmedia_index[:count_data])
        
        # итоговые данные для 2х графиков СМИ и Соцмедиа (bubbles)
        datas["list_socmedia"] = data["list_socmedia"]
        datas["list_socmedia_massage_count"] = data["list_socmedia_massage_count"]
        datas["list_socmedia_rating"] = [
            [int(x) for x in item] for item in data["list_socmedia_index"]]
        datas["filename"] = filenames
        
        second_graph_socmedia.append([{'name': x, 'count': y, 'rating': z} for x,y,z in zip(datas['list_socmedia'][0], 
                                                        datas['list_socmedia_massage_count'][0], datas['list_socmedia_rating'][0])])


    second_graph = []

    for i in range(len(filenames)):
        
        data = {}
        data['index_name'] = filenames[i]
        data['SMI'] =  second_graph_smi[i]
        data['Socmedia'] =  second_graph_socmedia[i]
        second_graph.append(data)


    #### график с динамикой отношения в СМИ и соцмедиа (динамика bubbles)
    # bubbles3 - график с динамикой отношения в СМИ и соцмедиа (динамика bubbles)
    date_bobble = []
    name_bobble = []
    index_bobble = []
    z_index_bobble = []
    tonality_index_bobble = []
    tonality_url = []

    count_date = 50
    third_graph_smi = []
    third_graph_socmedia = []

    for j in range(len(another_graph)):

        df = pd.DataFrame(another_graph[j])

        # метаданные
        columns = ['citeIndex', 'timeCreate', 'toneMark',
                   'hubtype', 'hub', 'audienceCount', 'url']
        # columns.remove('text')
        df_meta = pd.concat(
            [pd.DataFrame([x['authorObject'] if 'authorObject' in x else '' for x in another_graph[j]]),
             df[columns]], axis=1)
        # timestamp to date
        df_meta['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df_meta['timeCreate'].values]

        # индекс - время создания поста
        df_meta = df_meta.set_index(['timeCreate'])

        df_meta['timeCreate'] = list(df_meta.index)
        df_meta = df_meta[df_meta['hubtype'] == 'Онлайн-СМИ']

        # negative smi
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)][
            ['hub', 'citeIndex']].values

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
            Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)]['hub'])))

        fin_neg_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_neg, dict_neg_hubs_count):
            for key, value in d.items():
                fin_neg_dict[key] += (value,)

        list_neg_smi = list(fin_neg_dict.keys())
        list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
        list_neg_smi_massage_count = [x[1]
                                      for x in fin_neg_dict.values()]

        # positive smi
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)][
            ['hub', 'citeIndex']].values

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
            Counter(list(df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)]['hub'])))

        fin_pos_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_pos, dict_pos_hubs_count):
            for key, value in d.items():
                fin_pos_dict[key] += (value,)

        list_pos_smi = list(fin_pos_dict.keys())
        list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
        list_pos_smi_massage_count = [x[1]
                                      for x in fin_pos_dict.values()]

        # data to bobble graph
        bobble = []
        df_tonality = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
            ['hub', 'citeIndex', 'toneMark', 'url']].values
        index_ton = df_meta[(df_meta['hubtype'] == 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
            ['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1,
                                                                                                1)).total_seconds() * 1000)
                    for x in date_ton]

        for i in range(len(df_tonality)):
            if df_tonality[i][2] == -1:
                bobble.append(
                    [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
            elif df_tonality[i][2] == 1:
                bobble.append(
                    [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

        colors_red = ['#8B0000', '#FF4500', '#FFA07A']
        colors_green = ['#006400', '#00FF00', '#8FBC8F']

        for i in range(len(bobble)):
            if bobble[i][3] == 1:
                bobble[i][3] = colors_green[j]
            else:
                bobble[i][3] = colors_red[j]

        list_neg_smi = [x for x in list_neg_smi]
        list_pos_smi = [x for x in list_pos_smi]
        names_bobble = [x[1] for x in bobble]  # названия источников
        # названия источников
        names_bobble = [x for x in names_bobble]

        # count_date = 50  # сколько данных взять из списков
        date_bobble.append([x[0] for x in bobble][:count_date])
        name_bobble.append(names_bobble[:count_date])
        index_bobble.append([x[2] for x in bobble][:count_date])
        z_index_bobble.append([1] * len(bobble[:count_date]))
        tonality_index_bobble.append(
            [x[3] for x in bobble][:count_date])
        tonality_url.append([x[4] for x in bobble][:count_date])

        third_graph_smi.append([{'date': x, 'name': y, 'rating': z, 'url': h} for x,y,z,h in 
                              zip(date_bobble[0], name_bobble[0], index_bobble[0], tonality_url[0])])
        

    # bubbles4 - график с динамикой отношения в соцмедиа (динамика bubbles)
    date_bobble = []
    name_bobble = []
    index_bobble = []
    z_index_bobble = []
    tonality_index_bobble = []
    tonality_url = []

    for j in range(len(another_graph)):

        df = pd.DataFrame(another_graph[j])

        # метаданные
        columns = ['citeIndex', 'timeCreate', 'toneMark',
                   'hubtype', 'hub', 'audienceCount', 'url']
        # columns.remove('text')
        df_meta = pd.concat(
            [pd.DataFrame([x['authorObject'] if 'authorObject' in x else '' for x in another_graph[j]]),
             df[columns]], axis=1)
        # timestamp to date
        df_meta['timeCreate'] = [datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S') for x in
                                 df_meta['timeCreate'].values]

        # индекс - время создания поста
        df_meta = df_meta.set_index(['timeCreate'])

        df_meta['timeCreate'] = list(df_meta.index)
        df_meta = df_meta[df_meta['hubtype'] != 'Онлайн-СМИ']

        # negative Соцмедиа
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)][
            ['hub', 'audienceCount']].values

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
        list_neg = [int(x[0]) for x in list_neg]

        for i in range(len(list_neg)):
            dict_neg[list(dict_neg.keys())[i]] = list_neg[i]

        dict_neg = dict(
            sorted(dict_neg.items(), key=lambda x: x[1], reverse=True))

        dict_neg_hubs_count = dict(
            Counter(list(df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] == -1)]['hub'])))

        fin_neg_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_neg, dict_neg_hubs_count):
            for key, value in d.items():
                fin_neg_dict[key] += (value,)

        list_neg_smi = list(fin_neg_dict.keys())
        list_neg_smi_index = [x[0] for x in fin_neg_dict.values()]
        list_neg_smi_massage_count = [x[1]
                                      for x in fin_neg_dict.values()]

        # positive Соцмедиа
        df_hub_siteIndex = df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)][
            ['hub', 'audienceCount']].values

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
        list_pos = [int(x[0]) for x in list_pos]

        for i in range(len(list_pos)):
            dict_pos[list(dict_pos.keys())[i]] = list_pos[i]

        dict_pos = dict(
            sorted(dict_pos.items(), key=lambda x: x[1], reverse=True))

        dict_pos_hubs_count = dict(
            Counter(list(df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] == 1)]['hub'])))

        fin_pos_dict = defaultdict(tuple)
        # you can list as many input dicts as you want here
        for d in (dict_pos, dict_pos_hubs_count):
            for key, value in d.items():
                fin_pos_dict[key] += (value,)

        list_pos_smi = list(fin_pos_dict.keys())
        list_pos_smi_index = [x[0] for x in fin_pos_dict.values()]
        list_pos_smi_massage_count = [x[1]
                                      for x in fin_pos_dict.values()]

        # data to bobble graph
        bobble = []
        df_tonality = df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
            ['hub', 'audienceCount', 'toneMark', 'url']].values
        index_ton = df_meta[(df_meta['hubtype'] != 'Онлайн-СМИ') & (df_meta['toneMark'] != 0)][
            ['timeCreate']].values.tolist()
        date_ton = [x[0] for x in index_ton]
        date_ton = [int((datetime.strptime(x, '%Y-%m-%d %H:%M:%S') - datetime(1970, 1,
                                                                                                1)).total_seconds() * 1000)
                    for x in date_ton]

        for i in range(len(df_tonality)):
            if df_tonality[i][2] == -1:
                bobble.append(
                    [date_ton[i], df_tonality[i][0], dict_neg[df_tonality[i][0]], -1, df_tonality[i][4]])
            elif df_tonality[i][2] == 1:
                bobble.append(
                    [date_ton[i], df_tonality[i][0], dict_pos[df_tonality[i][0]], 1, df_tonality[i][4]])

        colors_red = ['#8B0000', '#FF4500', '#FFA07A']
        colors_green = ['#006400', '#00FF00', '#8FBC8F']

        for i in range(len(bobble)):
            if bobble[i][3] == 1:
                bobble[i][3] = colors_green[j]
            else:
                bobble[i][3] = colors_red[j]

        list_neg_smi = [x for x in list_neg_smi]
        list_pos_smi = [x for x in list_pos_smi]
        names_bobble = [x[1] for x in bobble]  # названия источников
        # названия источников
        names_bobble = [x for x in names_bobble]

        # count_date = 50  # сколько данных взять из списков
        date_bobble.append([x[0] for x in bobble][:count_date])
        name_bobble.append(names_bobble[:count_date])
        index_bobble.append([x[2] for x in bobble][:count_date])
        z_index_bobble.append([1] * len(bobble[:count_date]))
        tonality_index_bobble.append(
            [x[3] for x in bobble][:count_date])
        tonality_url.append([x[4] for x in bobble][:count_date])

        # финальные данные для графика динамики с конкурентами по Соцмедиа
        data_chart_4 = {"date_bobble": date_bobble,  # дата поста
                        "name_bobble": name_bobble,  # имя источника
                        "index_bobble": index_bobble,  # аудитория поста Соцмедиа
                        "z_index_bobble": z_index_bobble,
                        "tonality_index_bobble": tonality_index_bobble,  # цвет шаров
                        "tonality_url": tonality_url,  # ссылка на пост
                        "filenames": filenames}
        
        third_graph_socmedia.append([{'date': x, 'name': y, 'rating': z, 'url': h} for x,y,z,h in 
                              zip(date_bobble[0], name_bobble[0], index_bobble[0], tonality_url[0])])


    third_graph = []

    for i in range(len(filenames)):
        
        data = {}
        data['index_name'] = name_files[i]
        data['SMI'] =  third_graph_smi[i]
        data['Socmedia'] =  third_graph_socmedia[i]
        third_graph.append(data)

    return CompetitorsModel(first_graph=first_graph, second_graph=second_graph, third_graph=third_graph)


class DataFolder(BaseModel):
    name: str
    values: List[str]


class ModelDataFolder(BaseModel):
    values: List[DataFolder]

@app.get('/data-folders')
async def data_folders(user: User = Depends(current_user)) -> ModelDataFolder:

    es_indexes = [index for index in es.indices.get('*')] # список всех индексов elastic
    es_indexes = [x.strip() for x in es_indexes]

    if user.theme_rules == 'admin': # если пользователь админ, то вернуть все темы

        folders = '/home/dev/fastapi/analytics_app/data/json_files'
        sub_folders = [name for name in os.listdir(folders) if os.path.isdir(os.path.join(folders, name))]

        data_values = []
        os.chdir(folders)
        for i in range(len(sub_folders)):
            data_values.append({"name": sub_folders[i], 
                               "values": [f for f in listdir(sub_folders[i]) if isfile(join(sub_folders[i], f))]}) 
      
        return ModelDataFolder(values=data_values)
    
    else: # если пользователь не админ, то вернуть его темы
        data_index = []
        user_index = list(set(es_indexes) & set([x.strip().lower().replace('.json', '') for x in user.theme_rules.split(',')]))

        return ModelDataFolder(values=data_values)


@app.get('/create-folder')
async def create_folder(user: User = Depends(current_user), name: str = None):

    if name == None:
        return f'Укажите название папки.'
    
    os.chdir('/home/dev/fastapi/analytics_app/data/json_files')
    folders = '/home/dev/fastapi/analytics_app/data/json_files'
    sub_folders = [name for name in os.listdir(folders) if os.path.isdir(os.path.join(folders, name))]

    if name in sub_folders:
        return f'Папка с таким именем уже существует.'

    os.mkdir(name.strip())
    return f'Папка с именем {name} создана!'

# удаление файла или папки из сервиса
@app.get('/data-delete')
async def data_delete(folder_name: str = None, file_name: str = None):
    os.chdir('/home/dev/fastapi/analytics_app/data/json_files')
    if folder_name != None and file_name == None: # удаление папки
        shutil.rmtree('/home/dev/fastapi/analytics_app/data/json_files/' + folder_name, ignore_errors=True)
        return f'Папка {folder_name} удалена!'
    
    if folder_name != None and file_name != None: # удаление файла
        os.remove('/home/dev/fastapi/analytics_app/data/json_files/' + folder_name + 
                  '/' + file_name)
        return f'Файл {file_name} удален!'

@app.get('/file-rename')
async def rename(folder_name: str, current_file_name: str, new_file_name: str):
    os.chdir('/home/dev/fastapi/analytics_app/data/json_files/' + folder_name)
    os.rename(current_file_name, new_file_name)
    return f'Файл {current_file_name} переименован в {new_file_name}'


@app.post("/upload-file/")
async def create_upload_file(uploaded_file: UploadFile = File(...), folder_name: str = None):    
    file_location = '/home/dev/fastapi/analytics_app/data/json_files/' + folder_name + '/' + str(uploaded_file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(uploaded_file.file, file_object)
        
    return {"info": f"file '{uploaded_file.filename}' сохранен в папку '{folder_name}'"}


@app.get("/create-data-projector")
async def create_data_projector(folder_name: str, file_name: str):

    embed = hub.load("/home/dev/fastapi/analytics_app/data/embed_files/universal-sentence-encoder-multilingual_3")

    os.chdir('/home/dev/fastapi/analytics_app/data/json_files/' + folder_name)
    try:
        with io.open(file_name, encoding='utf-8', mode='r') as train_file:
            dict_train = json.load(train_file, strict=False)

    except:
        a = []
        with open(file_name, encoding='utf-8', mode='r') as file:
            for line in file:
                a.append(line)
        dict_train = []
        for i in range(len(a)):
            try:
                dict_train.append(ast.literal_eval(a[i]))
            except:
                continue
        dict_train = [x[0] for x in dict_train]

    df = pd.DataFrame(dict_train)

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

        if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
            df_meta = pd.concat([df_meta_socm, df_meta_smi])
        elif 'df_meta_smi' and 'df_meta_socm' not in locals():
            df_meta = df_meta_smi
        else:
            df_meta = df_meta_socm

    # тексты
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

    sent_ru = df_text['text'].values

    a = []
    for sent in sent_ru:
        # a.append(embed(sent)[0].np())
        a.append([np.round(x, 8) for x in embed(sent)[0].numpy()])

    embed_list = a

    dff = pd.DataFrame(a)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(dff.values)

    coord_list = []

    for i in range(len(x_tsne.tolist())):
        coord_list.append(', '.join([str(x) for x in x_tsne.tolist()[i]]))

    names = df_meta['fullname'].values.tolist()
    names = [x if x != '' else 'None' for x in names]

    regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    names_list = [words_only(x) if type(
        x) != float else 'None' for x in names]
    names_list = [preprocess_text(x) if type(
        x) != float else 'None' for x in names]
    names_list = [remove_stopwords(x) if type(
        x) != float else 'None' for x in names]
    names_list = ['None' if x == '' else x for x in names_list]

    name_str = '\n'.join(names_list)
    coord_list_str = '\n'.join(coord_list)

    path_project_data = '/home/dev/fastapi/analytics_app/data/projector_files/'
    os.chdir(path_project_data)
    try:
        os.mkdir(folder_name.replace('.json', ''))
    except:
        pass
    os.chdir(path_project_data + folder_name.replace('.json', ''))

    # сохранение данных для tsne
    dict_tsne = {}
    dict_tsne['author_name_str'] = name_str
    dict_tsne['coord_list_str'] = coord_list_str

    # with open(file_name.replace('.json', '') + '_projector.txt', 'w') as my_file:
    #     json.dump(dict_tsne, my_file)
    tsv_embed_list = []
    for i in range(len(embed_list)):
        tsv_embed_list.append('\t'.join([str(x) for x in embed_list[i]]))

    # test_names = names_list
    # with open(file_name.replace('.json', '.txt'), 'w', encoding="utf-8") as f:
    #     for line in test_names:
    #         f.write(f"{line}\n")

    os.chdir(path_project_data + folder_name)
    with open(file_name.replace('.json', '_author_point.tsv'), 'w') as f:
        for line in tsv_embed_list:
            f.write(f"{line}\n")
    test_names = names_list
    with open(file_name.replace('.json', '_author_name.txt'), 'w', encoding="utf-8") as f:
        for line in test_names:
            f.write(f"{line}\n")

    # # TSNE to unique authors
    # # подготовка данных .csv для TSNE уникальных авторов (тексты авторов складываются и убираются дубли)
    # print('Start Unique Author tsne emb')
    # os.chdir(path_to_files)
    # embed = hub.load(
    #     "/home/dev/social_app/data/universal-sentence-encoder-multilingual_3")

    # os.chdir(path_to_files + '/' + foldername)
    # try:
    #     with io.open(file_name, encoding='utf-8', mode='r') as train_file:
    #         dict_train = json.load(train_file, strict=False)

    # except:
    #     a = []
    #     with open(file_name, encoding='utf-8', mode='r') as file:
    #         for line in file:
    #             a.append(line)
    #     dict_train = []
    #     for i in range(len(a)):
    #         try:
    #             dict_train.append(ast.literal_eval(a[i]))
    #         except:
    #             continue
    #     dict_train = [x[0] for x in dict_train]

    # df = pd.DataFrame(dict_train)

    # # метаданные
    # # разбивка и сборка соцмедиа и СМИ в один датафрэйм с данными
    # df_meta = pd.DataFrame()

    # # случай выгрузки темы только по СМИ
    # if 'hubtype' not in df.columns:

    #     dff = df
    #     dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
    #         '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
    #     df_meta_smi_only = dff[[
    #         'timeCreate', 'hub', 'toneMark', 'audience', 'url', 'text', 'citeIndex']]
    #     # df_meta_smi_only.columns = ['timeCreate', 'hub', 'toneMark', 'audienceCount', 'url', 'text', 'citeIndex']
    #     df_meta_smi_only['fullname'] = dff['hub']
    #     df_meta_smi_only['author_type'] = 'Онлайн-СМИ'
    #     df_meta_smi_only['hubtype'] = 'Онлайн-СМИ'
    #     df_meta_smi_only['type'] = 'Онлайн-СМИ'
    #     df_meta_smi_only['er'] = 0
    # #     df_meta_smi_only = df_meta_smi_only[columns]

    #     df_meta = df_meta_smi_only

    # if 'hubtype' in df.columns:

    #     for i in range(2):  # Онлайн-СМИ или соцмедиа

    #         if i == 0:
    #             dff = df[df['hubtype'] != 'Онлайн-СМИ']
    #             if dff.shape[0] != 0:

    #                 dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
    #                     '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
    #                 df_meta_socm = dff[['timeCreate', 'hub', 'toneMark',
    #                                     'audienceCount', 'url', 'er', 'hubtype', 'text', 'type']]
    #                 df_meta_socm['fullname'] = pd.DataFrame.from_records(
    #                     dff['authorObject'].values)['fullname'].values
    #                 df_meta_socm['author_type'] = pd.DataFrame.from_records(
    #                     dff['authorObject'].values)['author_type'].values

    #         if i == 1:
    #             dff = df[df['hubtype'] == 'Онлайн-СМИ']
    #             if dff.shape[0] != 0:
    #                 dff['timeCreate'] = [datetime.fromtimestamp(x).strftime(
    #                     '%Y-%m-%d %H:%M:%S') for x in dff['timeCreate'].values]
    #                 df_meta_smi = dff[['timeCreate', 'hub', 'toneMark',
    #                                     'audienceCount', 'url', 'er', 'hubtype', 'text', 'citeIndex']]
    #                 df_meta_smi['fullname'] = dff['hub']
    #                 df_meta_smi['author_type'] = 'Онлайн-СМИ'
    #                 df_meta_smi['hubtype'] = 'Онлайн-СМИ'
    #                 df_meta_smi['type'] = 'Онлайн-СМИ'

    #     if 'df_meta_smi' in locals() and 'df_meta_socm' in locals():
    #         df_meta = pd.concat([df_meta_socm, df_meta_smi])
    #     elif 'df_meta_smi' and 'df_meta_socm' not in locals():
    #         df_meta = df_meta_smi
    #     else:
    #         df_meta = df_meta_socm

    # df_text = df[['text', 'authorObject', 'hub']]

    # # подготовка данных: текст и имя автора - замена имени автора если нет authorObject (это СМИ) - указывать hub (название СМИ)
    # a = []

    # for i in range(len(df_text['authorObject'].values)):
    #     try:
    #         a.append(df_text['authorObject'].values[i]['fullname'])
    #     except:
    #         a.append(df_text['hub'].values[i])

    # df_text['author_name'] = a
    # df_text = df_text[['author_name', 'text']]
    # # df_text.drop(['authorObject', 'hub'], axis=1, inplace=True)
    # # группируем в словарь автор: сообщения
    # a = {k: g["text"].tolist() for k, g in df_text.groupby("author_name")}
    # # убираем дубли сообщений из текстов автора
    # a = {k: ' '.join(list(set(v))) for (k, v) in a.items()}
    # # создаем финальный dataframe c автором и его уникальными текстами
    # df_text = pd.DataFrame(a.items())
    # df_text.columns = ['author_name', 'text']

    # regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    # def words_only(text, regex=regex):
    #     try:
    #         return " ".join(regex.findall(text))
    #     except:
    #         return ""

    # mystopwords = ['это', 'наш', 'тыс', 'млн', 'млрд', 'также', 'т', 'д', 'URL',
    #                 'i', 's', 'v', 'info', 'a', 'подробнее', 'который', 'год',
    #                 ' - ', '-', 'В', '—', '–', '-', 'в', 'который']

    # def preprocess_text(text):
    #     text = text.lower().replace("ё", "е")
    #     text = re.sub('((www\[^\s]+)|(https?://[^\s]+))', 'URL', text)
    #     text = re.sub('@[^\s]+', 'USER', text)
    #     text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    #     text = re.sub(' +', ' ', text)
    #     return text.strip()

    # def remove_stopwords(text, mystopwords=mystopwords):
    #     try:
    #         return " ".join([token for token in text.split() if not token in mystopwords])
    #     except:
    #         return ""

    # df_text['text'] = df_text['text'].apply(words_only)
    # df_text['text'] = df_text['text'].apply(preprocess_text)
    # df_text['text'] = df_text['text'].apply(remove_stopwords)

    # sent_ru = df_text['text'].values

    # a = []
    # for sent in sent_ru:
    #     # a.append(embed(sent)[0].np())
    #     a.append([np.round(x, 10) for x in embed(sent)[0].np()])

    # embed_list = a
    # dff = pd.DataFrame(a)

    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # x_tsne = tsne.fit_transform(dff.values)

    # coord_list = []
    # for i in range(len(x_tsne.tolist())):
    #     coord_list.append(', '.join([str(x) for x in x_tsne.tolist()[i]]))

    # names = df_text['author_name'].values.tolist()
    # names = [x if x != '' else 'None' for x in names]

    # regex = re.compile("[А-Яа-я:=!\)\()A-z\_\%/|]+")

    # names_list = [words_only(x) if type(
    #     x) != float else 'None' for x in names]
    # names_list = [preprocess_text(x) if type(
    #     x) != float else 'None' for x in names]
    # names_list = [remove_stopwords(x) if type(
    #     x) != float else 'None' for x in names]
    # names_list = ['None' if x == '' else x for x in names_list]

    # name_str = '\n'.join(names_list)
    # coord_list_str = '\n'.join(coord_list)

    # path_tsne_data_unique_authors = path_tsne_data + '/unique_authors'
    # os.chdir(path_tsne_data_unique_authors)
    # # try:
    # #     os.mkdir(file_name.replace('.json', ''))
    # # except:
    # #     pass
    # # os.chdir(path_tsne_data_unique_authors + '/' + file_name.replace('.json', ''))

    # # сохранение данных для tsne
    # dict_tsne = {}
    # dict_tsne['author_name_str'] = name_str
    # dict_tsne['coord_list_str'] = coord_list_str

    # with open(file_name.replace('.json', '') + '_data_tsne.txt', 'w') as my_file:
    #     json.dump(dict_tsne, my_file)

    # del dict_tsne
    return f'yes'

if __name__ == "__main__":
    uvicorn.run("main:app", host="194.146.113.124", port=8005, reload=True)