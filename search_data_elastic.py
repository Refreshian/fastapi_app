from elasticsearch import Elasticsearch, helpers
import sys, json, os

es = Elasticsearch(
    ['localhost'],
    port=9200
)


def elastic_query(theme_index: str, query_str: str):
        
        # запрос всех имеющихся данных
        if query_str == 'all':
          query = {
              "size" : 10000,
              "query": {
                  "match_all": {}
              }
          }
          
          # data = [item for sublist in data for item in sublist]
          data = []
          data.append(es.search(index=theme_index, body=query)['hits']['hits'])
          data = [item for sublist in data for item in sublist]
          data = [x['_source'] for x in data] # получение финального фрейма данных от elastic

        else:

          data = []

          query_str = query_str.split(',')
          query_str = [x.strip() for x in query_str]

          for i in range(len(query_str)):

              # запрос слов с морфологией "query": "сын | и | отец | стань аналитиком"
              if ' или' in query_str[i]:
                  print("или")
                  query = {
                    "size" : 10000,
                    "query": {
                      "query_string": {
                        "query": query_str[i].replace(' или', ' |'),
                        "default_field": "text"
                      }
                    }
                  }
                  data.append(es.search(index=theme_index, body=query)['hits'])

              # поиск с присутствием всех перечисленных слов в документах
              elif " и" in query_str[i]:
                  print('NO')
                  query = {
                  "size" : 10000,
                  "query": {
                    "query_string": {
                      "query": query_str[i].replace(' и', ' AND'),
                      "default_field": "text"
                    }
                  }
                  }
                  data.append(es.search(index=theme_index, body=query)['hits'])

              # поиск с минус-критерием
              elif " -" in query_str[i]:
                  q = query_str[i].split(' -')
                  query = {
                      "size" : 10000,
                      "query": {
                        "bool": {
                            "must_not": [
                              {"match_phrase":{"text":q[1]}}
                            ], 
                            "should": [
                                        {"match_phrase":{"text":q[0]}}
                            ], "minimum_should_match":1 
                        }
                    }
                  }
                  data.append(es.search(index=theme_index, body=query)['hits'])

              # Поиск с расстоянием между слов во фразе
              elif '~' in query_str[i]:
                  q = query_str[i].split('~')
                  query = {
                      "size" : 10000,
                      "query":
                    {"match_phrase":
                      {"text": 
                        {"query": q[0], "slop":q[1]}}
                    }
                  }
                  data.append(es.search(index=theme_index, body=query)['hits'])                

              # запрос фразы с морфологией "query": "аналитика данных"
              else:
                  query = {
                      "size" : 10000,
                    "query": {
                      "match_phrase": {
                        "text": {
                          "query": query_str[i]
                        }
                      }
                    }
                  }
                  data.append(es.search(index=theme_index, body=query)['hits'])

          try:
            data = [x['hits'] for x in data]
            data = [item for sublist in data for item in sublist]
            data = [x['_source'] for x in data] # получение финального фрейма данных от elastic
          except:
            pass
        
        return data

