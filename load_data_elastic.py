import time
import json
from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch, helpers
import sys, json, os


es = Elasticsearch(
    ['194.146.113.124'],
    port=9200
)

# список имеющихся индексов
es_indexes = [index for index in es.indices.get('*')]
# es.indices.delete(index='read_me', ignore=[400, 404])

def load_file_to_elstic(filename, path=None):
    
    mapping = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "russian"
                },
                "text": {
                    "type": "text",
                    "analyzer": "russian"
                }
            }
        }
    }

    os.chdir(path)
    new_index = filename.filename.replace('.json', '').lower()
    print(new_index)
    # удаляем предыдущий идекс если загружается файл за те же даты для создания далее нового индекса
    if new_index in es_indexes:
        es.indices.delete(index=new_index, ignore=[400, 404])

    response = es.indices.create(
    index=new_index,
    body=mapping,
        ignore=400 # ignore 400 already exists code
    )

    print("!!!+++++!!!")

    if 'acknowledged' in response:
        if response['acknowledged'] == True:
            print("INDEX MAPPING SUCCESS FOR INDEX:", response['index'])

    # catch API error response
    elif 'error' in response:
        print("ERROR:", response['error']['root_cause'])
        print("TYPE:", response['error']['type'])

    # print out the response:
    print ('\nresponse:', response)

    # Elastic configuration.
    ELASTIC_ADDRESS = "http://localhost:9200"
    INDEX_NAME = new_index
    documents = []

    def index_documents(documents_filename, index_name, es_client):

        index = 0
        # Open the file containing the JSON data to index.
        with open(documents_filename.filename, "r") as json_file:
            json_data = json.load(json_file)
            # проставляем [] в geoObject для корректной загрузки в es
            for i in range(len(json_data)):
                if 'geoObject' in json_data[i]:
                    if not json_data[i]['geoObject'] == []:
                        json_data[i]['geoObject'] = []

            for doc in json_data:
                doc["_id"] = index
                
                documents.append(doc)
                index = index + 1


            # How you'd index data to Elastic.
            indexing = bulk(es_client, documents, index=index_name, chunk_size=100)
            print("Success - %s , Failed - %s" % (indexing[0], len(indexing[1])))
    
    index_documents(filename, INDEX_NAME, es)
    
# удаление индекса
# es.indices.delete(index='poc2', ignore=[400,404])