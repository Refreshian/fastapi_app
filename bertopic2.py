import os

print("Размер файла:", os.path.getsize('/home/dev/fastapi/analytics_app/data/html_files/topic_model_Platon_22.11.2024-21.12.2024') / (1024 * 1024), "Мегабайт")


# import tarfile
# from bertopic import BERTopic
# import io

# # Имя tar.gz файла
# tar_file_name = 'model_archive.tar.gz'
# model_file_name = 'topic_model_Platon_22.11.2024-21.12.2024'  # Название файла модели

# # Открытие tar.gz архива
# with tarfile.open(tar_file_name, 'r:gz') as tar_file:
#     # Проверяем наличие нужного файла модели
#     if model_file_name in tar_file.getnames():
#         # Извлекаем файл в память
#         member = tar_file.extractfile(model_file_name)

#         if member is not None:
#             # Читаем содержимое файла
#             model_content = member.read()  # Содержимое сохраняется в переменную

#             # Создаем временный байтовый поток для загрузки модели
#             model_io = io.BytesIO(model_content)

#             # Загружаем модель из байтового потока
#             topic_model = BERTopic.load(model_io)

#             print("Модель успешно загружена в переменную topic_model.")
#         else:
#             print(f'Не удалось извлечь файл {model_file_name}.')
#     else:
#         print(f'Файл {model_file_name} не найден в архиве.')