from elasticsearch import Elasticsearch, helpers
import sys, json, os

es = Elasticsearch(
    ['localhost'],
    port=9200
)

es.indices.put_settings(index="cennosti_data_year_without_doubles",
                        body= {"index" : {
                                "max_result_window" : 2000000
                              }})

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




# indexes = {1: "rosbank_01.02.2024-07.02.2024", 2: "skillfactory_zaprosy_na_obuchenie_15.01.2024-21.01.2024", 3:'rosbank_19.02.2024-29.02.2024', 
#            4: "rosbank_14.03.2024-14.03.2024_fullday", 5: "r_13.03.2024-14.03.2024_full", 6: "rosbank_22.03.2024-24.03.2024", 
#            7: "monitoring_tem_19.03.2024-25.03.2024", 8: 'rosbank_26.03.2024-01.04.2024', 9: 'tehfob', 10: 'transport_01.01.2024-09.04.2024', 
#            11: 'moskovskiy_transport_01.01.2024_09.04.2024_2b', 12: 'rosbank_01.04.2024-15.04.2024', 13: 'rosbank_14.05.2024-16.05_чистая прибыль',
#            14: 'contented_smi_01.04.2024-26.05.2024', 15: 'skillbox_smi_01.04.2024-26.05.2024', 16: 'rb_smi', 17: 'geekbrains', 18: 'eduson', 
#            19: 'maley_nlmk_boevaya_tema_17.06.2024-21.06.2024_66757eb24cb15033866ecdd8', 20: 'maley_nlmk_boevaya_tema_17_06_2024_21_06_2024',
#            21: 'platon_test_31.07.2024-06.08.2024', 22: 'platon_test', 23: 'avtomobili_01.09.2023-02.09.2024', 24: 'cennosti_01.08.2024-31.08.2024'}

# es.indices.put_settings(index="cennosti_01.08.2024-31.08.2024",
#                         body= {"index" : {
#                                 "max_result_window" : 500000
#                               }})

# data = elastic_query(theme_index=indexes[24], query_str='all')
# texts = [x['text'] for x in data]
# print(len(texts))