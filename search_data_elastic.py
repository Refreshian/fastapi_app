# from elasticsearch import Elasticsearch, helpers
# import sys, json, os

# es = Elasticsearch(
#     ['localhost'],
#     port=9200
# )


# # Функция для обновления настроек индекса
# def update_max_result_window(index_name: str, max_window: int = 2000000):
#     try:
#         es.indices.put_settings(
#             index=index_name,
#             body={
#                 "index": {
#                     "max_result_window": max_window
#                 }
#             }
#         )
#         # print(f"Настройки индекса '{index_name}' обновлены: max_result_window = {max_window}")
#     except Exception as e:
#         print(f"Ошибка при обновлении настроек индекса '{index_name}': {e}")

# def elastic_query(theme_index: str, query_str: str):
#     # Обновляем настройку max_result_window для текущего индекса
#     update_max_result_window(theme_index)
    
#     # Запрос всех имеющихся данных
#     if query_str == 'all':
#         query = {
#             "size": 1000000,
#             "query": {
#                 "match_all": {}
#             }
#         }
        
#         data = []
#         data.append(es.search(index=theme_index, body=query)['hits']['hits'])
#         data = [item for sublist in data for item in sublist]
#         data = [x['_source'] for x in data]  # Получение финального фрейма данных от Elastic

#     else:
#         data = []

#         query_str = query_str.split(',')
#         query_str = [x.strip() for x in query_str]

#         for i in range(len(query_str)):

#             # Запрос слов с морфологией "query": "сын | и | отец | стань аналитиком"
#             if ' или' in query_str[i]:
#                 print("или")
#                 query = {
#                     "size": 1000000,
#                     "query": {
#                         "query_string": {
#                             "query": query_str[i].replace(' или', ' |'),
#                             "default_field": "text"
#                         }
#                     }
#                 }
#                 data.append(es.search(index=theme_index, body=query)['hits'])

#             # Поиск с присутствием всех перечисленных слов в документах
#             elif " и" in query_str[i]:
#                 print('NO')
#                 query = {
#                     "size": 1000000,
#                     "query": {
#                         "query_string": {
#                             "query": query_str[i].replace(' и', ' AND'),
#                             "default_field": "text"
#                         }
#                     }
#                 }
#                 data.append(es.search(index=theme_index, body=query)['hits'])

#             # Поиск с минус-критерием
#             elif " -" in query_str[i]:
#                 q = query_str[i].split(' -')
#                 query = {
#                     "size": 1000000,
#                     "query": {
#                         "bool": {
#                             "must_not": [
#                                 {"match_phrase": {"text": q[1]}}
#                             ],
#                             "should": [
#                                 {"match_phrase": {"text": q[0]}}
#                             ],
#                             "minimum_should_match": 1
#                         }
#                     }
#                 }
#                 data.append(es.search(index=theme_index, body=query)['hits'])

#             # Поиск с расстоянием между словами во фразе
#             elif '~' in query_str[i]:
#                 q = query_str[i].split('~')
#                 query = {
#                     "size": 1000000,
#                     "query": {
#                         "match_phrase": {
#                             "text": {
#                                 "query": q[0],
#                                 "slop": int(q[1])
#                             }
#                         }
#                     }
#                 }
#                 data.append(es.search(index=theme_index, body=query)['hits'])                

#             # Запрос фразы с морфологией "query": "аналитика данных"
#             else:
#                 query = {
#                     "size": 1000000,
#                     "query": {
#                         "match_phrase": {
#                             "text": {
#                                 "query": query_str[i]
#                             }
#                         }
#                     }
#                 }
#                 data.append(es.search(index=theme_index, body=query)['hits'])

#         try:
#             data = [x['hits'] for x in data]
#             data = [item for sublist in data for item in sublist]
#             data = [x['_source'] for x in data]  # Получение финального фрейма данных от Elastic
#         except Exception as e:
#             print(f"Ошибка при обработке данных: {e}")
    
#     return data

from elasticsearch import Elasticsearch, helpers
import sys, json, os

es = Elasticsearch(
    ['localhost'],
    port=9200
)

# Функция для обновления настроек индекса
def update_max_result_window(index_name: str, max_window: int = 2000000):
    try:
        es.indices.put_settings(
            index=index_name,
            body={
                "index": {
                    "max_result_window": max_window
                }
            }
        )
    except Exception as e:
        print(f"Ошибка при обновлении настроек индекса '{index_name}': {e}")


# Обновленная функция elastic_query с использованием Scroll API
def elastic_query(theme_index: str, query_str: str, scroll_time: str = '5m', batch_size: int = 10000):
    # Обновляем настройку max_result_window для текущего индекса
    update_max_result_window(theme_index)
    
    # Стартовый запрос
    if query_str == 'all' or query_str  == None:
        query = {
            "query": {
                "match_all": {}
            }
        }
    else:
        # Для более сложных запросов из вашего кода (или | и ...)
        query = {
            "query": {
                "query_string": {
                    "query": query_str,
                    "default_field": "text"
                }
            }
        }

    # Получение первого набора данных и scroll_id
    try:
        response = es.search(
            index=theme_index,
            body=query,
            scroll=scroll_time,  # Время, которое контекст scroll будет "держаться"
            size=batch_size      # Размер одной пачки результатов
        )
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return []

    scroll_id = response['_scroll_id']  # Получение ID для Scroll-запроса
    total_hits = response['hits']['total']['value']  # Общее количество доступных записей
    results = response['hits']['hits']  # Начальная партия данных

    print(f"Общее количество документов: {total_hits}")

    # Забираем остальные данные в цикле
    while len(response['hits']['hits']) > 0:
        try:
            response = es.scroll(
                scroll_id=scroll_id,  # Используем предыдущий scroll_id
                scroll=scroll_time    # Продлеваем время жизни Scroll-контекста
            )
            results.extend(response['hits']['hits'])  # Добавляем новые данные
            scroll_id = response['_scroll_id']  # Обновляем scroll_id
        except Exception as e:
            print(f"Ошибка при выполнении scroll-запроса: {e}")
            break

    # Освобождаем контекст scroll
    try:
        es.clear_scroll(scroll_id=scroll_id)
    except:
        pass

    # Преобразуем результаты в нужный формат (_source из каждого документа)
    data = [hit['_source'] for hit in results]
    return data