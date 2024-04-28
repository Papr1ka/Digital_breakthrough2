import os
import torch
import torch.nn as nn
from transformers import BertTokenizerFast as BertTokenizer, BertModel
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
import logging
from os import path
from .models import Lesson
from django.conf import settings
from .charts import contains_url, contains_word, find_question_sentences
from urllib.parse import urlparse

from collections import Counter
from catboost import CatBoostRegressor

from math import floor

DATA = settings.DATA_FOLDER

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("loading model")


MODEL_BIN_PATH = path.abspath(path.join(DATA, 'best_model_state.bin'))
MODEL_HAPPY_BIN = path.abspath(path.join(DATA, 'best_model_happy.bin'))
PRE_TRAINED_MODEL_NAME = 'DeepPavlov/rubert-base-cased'
GUF = path.join(DATA, 'model-q4_K.gguf')
FINAL = path.join(DATA, 'final')

regressor_model = CatBoostRegressor()
regressor_model.load_model(FINAL, format="cbm")

weights = torch.load(MODEL_BIN_PATH, map_location=device)
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=5).to(device)
model.load_state_dict(weights)
model.eval()
tags_id_to_str = {
    0 : "бесполезные вещи",
    1 : "технические неполадки",
    2 : "вопросы",
    3 : "начало урока",
    4 : "конец урока"
}

logging.info('Model loaded')


model2 = None
tokenizer2 = None
weights2 = torch.load(MODEL_HAPPY_BIN, map_location=device)
tokenizer2 = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model2 = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=3).to(device)
model2.load_state_dict(weights2)
model2.eval()
tags_id_to_str2 = {
    0 : "нейтральное",
    1 : "позитивное",
    2 : "негативное"
}

def check_url(message):
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    matches = re.findall(pattern, message)
    domains = [urlparse(match).netloc for match in matches]
    if len(domains) == 1:
        return ', '.join(domains)
    else:
            return ''

def predict_text(lid):
    lesson = Lesson.objects.get(_id=lid)
    messages = lesson.messages()
    data = [message.content for message in messages]
    logging.error("Starting")
    batch = tokenizer(data, max_length=512, padding=True, truncation=True, return_tensors='pt')
    logging.error("barch end")
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    logging.error(str(len(data)))
    logging.error("start outputs")
    outputs = model(
    input_ids = input_ids,
    token_type_ids = token_type_ids,
    attention_mask = attention_mask
    )
    logging.error("start max")
    _, preds = torch.max(outputs.logits, dim=1)
    logging.error("end max")
    class_id = _.argmax().item()
    score = _[class_id].item()
    category_name = []
    for i in preds:
        category_name.append(tags_id_to_str[i.item()])
    
    for message, category_name in zip(messages, category_name):
        message.tag_bert1 = category_name
        message.save()
    
    lesson.handled_bert = True
    lesson.save(force_update=True)
    return True


"""GGUF версия Сайги3
Файлы для установки!!!

!pip install llama-cpp-python

!wget https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q4_K.gguf
"""
from llama_cpp import Llama
import pandas as pd
import re


def get_model_saiga(path='model-q4_K.gguf', n_ctx=1024):  # n_ctx - Контекстное окно, параметр используется и при генерации!
    llm = Llama(model_path=GUF, n_ctx=n_ctx, n_batch=126)  # Загрузка модели
    return llm

model_llvm = get_model_saiga()

def generate_resume_comments(comments, model=model_llvm, list_comments=False, n_ctx=1024):  # Генерация текста по комментариям, comments - строка объединенных через \n комментариев
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


def parse_saiga_text(text):  # Парсинг текста. text - текст, генерируемый Сайгой

    def get_part(pattern, id):
        res = re.findall(pattern, text)
        if res:
            return '.'.join(res)
        else:
            return 'нет'
    return {'Насколько часто ученики отвлекались': get_part(r'Ученики отвлекались - (\w+)', text),
            # Нахождение частоты отвлечения учеников
            'Использованные термины': get_part(r'\d+\.\s(.+?) \(\d+\)', text),  # Использованных технических терминов
            'Насколько легко было воспринимать материал': get_part(r'Оценка понимания - (.+)', text),  # Оценки понимания материала
            'Тема обсуждений': get_part(r'Главная тема - (\w+)', text),  # Темы обсуждения
            'Выявленная проблема': get_part(r'\d+\.\s(.*?\.)(?=\n\d+\.|\Z)', text)}


def predict_text2(lid):
    lesson = Lesson.objects.get(_id=lid)
    messages = lesson.messages()
    data = [message.content for message in messages]
    logging.error("Starting")
    batch = tokenizer2(data, max_length=512, padding=True, truncation=True, return_tensors='pt')
    logging.error("barch end")
    input_ids = batch["input_ids"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    logging.error(str(len(data)))
    logging.error("start outputs")
    outputs = model2(
    input_ids = input_ids,
    token_type_ids = token_type_ids,
    attention_mask = attention_mask
    )
    logging.error("start max")
    _, preds = torch.max(outputs.logits, dim=1)
    logging.error("end max")
    class_id = _.argmax().item()
    score = _[class_id].item()
    category_name = []
    for i in preds:
        category_name.append(tags_id_to_str2[i.item()])
    
    for message, category_name in zip(messages, category_name):
        message.tag_bert2 = category_name
        message.save()
    
    lesson.handled_bert_emo = True
    lesson.save(force_update=True)
    return True








"""Необходимо загрузить
!pip install catboost

От анализа ссылок Сони:
contains_word
contains_url
contains_questions
classify
check_url

От gguf_сайга:
parse_saiga_text

От скользящее окно Саши
SMA
anomaly_detection

От обработки сообщений Тимура:
pipeline_get_met
"""


def classify(domain):
    bad_keywords = ['garticphone', 'drawize', 'battleship']
    if not domain:
        return ''

    if any(keyword in domain.lower() for keyword in bad_keywords):
        return 'Irrelevant'
    else:
        return 'Relevant'


def get_regressor(path='final'):  # Загрузка модели
    model = CatBoostRegressor()
    model.load_model(path, format="cbm")
    return model


def predict_regressor(model, df_inp):  # Получение оценки
    computed_metrics = compute_metrics(df_inp)
    return model.predict(computed_metrics)

def pipeline_get_met(dt, preprocessed_df=None, plot_metrics=True) -> dict:
    '''
    
    расчет метрик
    с препроцессингом
    return dict
    
    '''
    metrics = {
        "Длина сообщения": -1,
        "avg_time": -1,
        "max_time": -1
    }

    df_merged_1 = dt.sort_values(['ID урока', 'Дата сообщения_x'])
    df_merged_1['time_diff'] = df_merged_1.groupby('ID урока')['Дата сообщения_x'].diff()
    df_merged_1['time_diff'] = df_merged_1['time_diff'].dt.total_seconds() / 60  

    average_time_diff_per_lesson = df_merged_1.groupby('ID урока')['time_diff'].mean()
    max_time_diff_per_lesson = df_merged_1.groupby('ID урока')['time_diff'].max()

    average_time_diff_per_lesson.name = "avg_time"
    max_time_diff_per_lesson.name = "max_time"
    

    try:
        metrics['avg_time'] = floor(average_time_diff_per_lesson.iloc[0]) if not pd.isna(average_time_diff_per_lesson.iloc[0]) else 0
    except Exception as E:
        metrics['avg_time'] = 0
    try:
        metrics['max_time'] = floor(avg_message_length.iloc[0]) if not pd.isna(max_time_diff_per_lesson.iloc[0]) else 0
    except Exception as E:
        metrics['max_time'] = 0
    
    dt['Длина сообщения'] = dt['Текст сообщения'].apply(lambda x: len(x.split()))
    avg_message_length = dt.groupby('ID урока')['Длина сообщения'].mean()
    try:
        metrics['Длина сообщения'] = floor(avg_message_length.iloc[0]) if not pd.isna(avg_message_length.iloc[0]) else 0
    except Exception as E:
        metrics['Длина сообщения'] = 0
    
    return metrics


def cm(data, saiga_text):  # Подсчет метрик для одного урока
    # data - df из всех сообщений одного урока + ['Текст от Сайги'], ['Тэги на тэги'], ['Тэги на эмоции']
    
    def SMA(arr):
        temp = []
        for i in range(len(arr)):
            temp.append(sum(arr[max(0, i-10):i]) / 10)
        return temp

    def anomaly_detection(arrayMA):
        bool_vec = []
        for i in range(1, len(arrayMA)):
            if 1.2 * arrayMA[i - 1] > arrayMA[i]:
                bool_vec.append(1)
            elif 1.4 * arrayMA[i - 1] > arrayMA[i]:
                bool_vec.append(2)
            else:
                bool_vec.append(0)
        return bool_vec
    
    comments = data["Текст сообщения"].tolist()

    data = contains_word(data)
    data = contains_url(data)
    data = find_question_sentences(data)

    df_inp = pd.Series()

    def count_tf(data, metric):
        n = data.iloc[:, metric].value_counts()
        if True in n:
            return n[True]
        else:
            return 0

    df_inp["Количество ругани"] = count_tf(data, -3)
    df_inp["Количество ссылок"] = count_tf(data, -2)
    df_inp["Количество вопросов"] = count_tf(data, -1)

    df_inp['Количество сообщений'] = data.shape[0]

    saiga_metrics = parse_saiga_text(saiga_text)
    for k, v in saiga_metrics.items():
      df_inp[k] = v
    
    def to_numt1(tag):
        return {'бесполезные вещи': 0,
                'технические неполадки': 1,
                'вопросы': 2,
                'начало урока': 3,
                'конец урока': 4}.get(tag)

    def to_numt2(tag):
        return {
            "нейтральное" : 0,
            "позитивное" : 1,
            "негативное" : 2
        }.get(tag)
    
    data['Тэги на тэги'] = data['Тэги на тэги'].apply(to_numt1)
    data['Тэги на эмоции'] = data['Тэги на эмоции'].apply(to_numt2)
    
    tags = data['Тэги на тэги']

    def most_rag(tags):
        tags_str_to_id = {
            0: "бесполезные вещи",
            1: "технические неполадки",
            2: "вопросы",
            3: "начало урока",
            4: "конец урока"
        }

        counter = Counter(tags)

        return counter.most_common(1)[0][0]

    df_inp['Преобладающий тег'] = tags_id_to_str.get(most_rag(tags))

    def check_susp(tags):
        arr = SMA(tags)
        bool_vec = anomaly_detection(arr)
        if bool_vec.count(2) > 8:
            return True
        else:
            return False

    df_inp['Была ли подозрительная активность'] = check_susp(tags)

    em_tags = data['Тэги на эмоции']

    def find_common_emotion(em_tags):
        counter = Counter(em_tags)

        return counter.most_common(1)[0][0]

    df_inp['Преобладающая эмоция'] = 0

    rel_data = list(map(check_url, comments))
    rel_data = list(map(classify, rel_data))
    counter = Counter(rel_data)
    
    
    ref_type = counter.most_common(1)[0][0]
    if ref_type is None:
        ref_type = 0
    df_inp['Тип ссылок'] = ref_type

    id = data.iloc[0]["ID урока"]
    mes_metrics = pipeline_get_met(data)
    for k, v in mes_metrics.items():
        df_inp[k] = v

    return df_inp

def predict_regressor(df_inp, saiga_text):  # Получение оценки
    computed_metrics = cm(df_inp, saiga_text).fillna(0)
    computed_metrics = computed_metrics.drop(['Выявленная проблема'])
    print(computed_metrics)
    return regressor_model.predict(computed_metrics)


def generate_final_grade(df_input, model=model_llvm,
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
    print(df_input)
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
