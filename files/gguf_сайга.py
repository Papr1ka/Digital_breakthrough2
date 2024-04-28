"""GGUF версия Сайги3
Файлы для установки!!!

!pip install llama-cpp-python

!wget https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf
"""
from llama_cpp import Llama
import pandas as pd
import re


def get_model_saiga(path='model-q4_K.gguf',
                    n_ctx=1024):  # n_ctx - Контекстное окно, параметр используется и при генерации!
    llm = Llama(path, n_ctx=n_ctx, n_batch=126)  # Загрузка модели
    return llm


def generate_resume_comments(comments, model, list_comments=False,
                             n_ctx=1024):  # Генерация текста по комментариям, comments - строка объединенных через \n комментариев
    # Генерация: 2-3 минуты
    # Формат возвращаемой строки
    # assustant\n\n
    # Главная тема -...Ученики отвлекались - ... Использованные технические термины: ... Оценка понимания - ... Проблемы урока: ...

    if list_comments:
        comments = '\n'.join(comments)

    task = "\nДаны комментарии учеников к видео-уроку. Давай ответы по шаблонам. Для маркировки списков используй символ '-'. \
    Выдели главную тему обсуждений. Шаблон: Главная тема - тема. \
    Определи, насколько часто обсуждения были не по теме урока: часто, не очень часто, редко. \
    Шаблон: Ученики отвлекались - частота. \
    Дай список используемых технических терминов и количество их употреблений. \
    Оцени, насколько тяжело было ученикам воспринимать урок: тяжело, не очень тяжело, легко. \
    Шаблон: Оценка понимания - оценка.\
    Дай список проблем урока, не более 5."  # Часть пользовательского промпта

    system = "Ты — русскоязычный автоматический ассистент преподавателя. \
        Ты анализируешь комментарии учеников видео-урока."  # Системный промпт

    text = task + comments  # Объединение промпта и комментариев
    text = text[:n_ctx]  # Обрезка для контекстного окна!!

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>{text}<|eot_id|>"""  # Создание промпта по формату для Сайги

    output = model(prompt, temperature=0.2, top_p=0.5, echo=False, max_tokens=3584)
    output_text = output["choices"][0]["text"].strip()

    return output_text


def generate_final_grade(df_input, model,
                         n_ctx=1024):  # Генерация финальной оценки урока, df_input - Series с метриками
    # Промпты
    task = "\nДаны метрики оценки комментариев видео-урока.\
      Оцени качество преподавания как высокое, среднее, удовлетворительное, плохое.\
      Выяви 3 метрики из данных, от которых больше всего зависит качество преподавания. Объясни, почему. \
      Составь общее резюме урока.\
      Дай ответ по шаблону. \
      Шаблон: Оценка качества преподавания, Важные оценочные метрики (объяснение их значимости), Резюме."  # Пользовательский для шаблонной генерации

    system = "Ты — русскоязычный автоматический ассистент преподавателя. \
            Ты анализируешь комментарии учеников видео-урока."  # Системный

    # Обработка метрик
    df_input = df_input.rename(
        {"avg_time": "Средняя пауза между сообщениями", "max_time": "Максимальная пауза между сообщениями",
                 "Длина сообщения": "Средняя длина сообщений"})

    def ch_url(x):
        if x == "Relevant":
            return "Относились к теме"
        elif x == "Irrelevant":
            return "Не относились к теме"
        else:
            return 'Не было'

    def ch_em(x):
        if x == 0:
            return 'Нейтральная'
        elif x == 1:
            return 'Положительная'
        else:
            return 'Негативная'

    df_input['Тип ссылок'] = ch_url(df_input['Тип ссылок'])
    df_input['Была ли подозрительная активность'] = "Была" if df_input['Была ли подозрительная активность'] else "Не была"
    df_input['Преобладающая эмоция'] = ch_em(df_input['Преобладающая эмоция'])

    metrics = df_input.to_dict()  # Первые 3 поля отбрасываются, потому что это 2 фиктивных и ID урока
    for k, v in metrics.items():
        task += f'{k}:{v}.'

    # Генерация текста
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>
           <|start_header_id|>user<|end_header_id|>{task}<|eot_id|>"""  # Создание промпта по формату для Сайги

    output = model(prompt, temperature=0.2, top_p=0.5, echo=False, max_tokens=3584)
    output_text = output["choices"][0]["text"].strip()

    return output_text


def parse_saiga_text(text):  # Парсинг текста. text - текст, генерируемый Сайгой

    def get_part(pattern, id):
        res = re.findall(pattern, text)
        if res:
            return '.'.join(res)
        else:
            return 'нет'

    return {'Насколько часто отвлекались': get_part(r'Ученики отвлекались - (\w+)', text),
            # Нахождение частоты отвлечения учеников
            'Использование терминов': get_part(r'\d+\.\s(.+?) \(\d+\)', text),  # Использованных технических терминов
            'Насколько понятен материал': get_part(r'Оценка понимания - (.+)', text),  # Оценки понимания материала
            'Тема обсуждений': get_part(r'Главная тема - (\w+)', text),  # Темы обсуждения
            'Список проблем': get_part(r'\d+\.\s(.*?\.)(?=\n\d+\.|\Z)', text)}  # Списка проблем
