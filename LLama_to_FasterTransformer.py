
# Загрузка словаря истории запросов пользователей
import os
import pickle
from datetime import datetime
import numpy as np

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

# Загружаем данные
# Путь к файлу с темами 
file_path = '/home/dev/fastapi/analytics_app/data/indexes.pkl'
# Загрузка словаря с темами
indexes = load_dict_from_pickle(file_path)

print(indexes)



