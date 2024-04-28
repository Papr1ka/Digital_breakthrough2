from .charts import contains_url, contains_word, find_question_sentences
import re
from urllib.parse import urlparse
import pandas as pd
from collections import Counter


tags_id_to_str = {
    0 : "бесполезные вещи",
    1 : "технические неполадки",
    2 : "вопросы",
    3 : "начало урока",
    4 : "конец урока"
}

tags_id_to_str2 = {
    0 : "нейтральное",
    1 : "позитивное",
    2 : "негативное"
}


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


def check_url(message):
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

    matches = re.findall(pattern, message)
    domains = [urlparse(match).netloc for match in matches]
    if len(domains) == 1:
        return ', '.join(domains)
    else:
            return ''


def classify(domain):
    bad_keywords = ['garticphone', 'drawize', 'battleship']
    if not domain:
        return ''

    if any(keyword in domain.lower() for keyword in bad_keywords):
        return 'Irrelevant'
    else:
        return 'Relevant'

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
        metrics['avg_time'] = average_time_diff_per_lesson.iloc[0]
    except ValueError:
        metrics['avg_time'] = 0
    try:
        metrics['max_time'] = max_time_diff_per_lesson.iloc[0]
    except ValueError:
        metrics['max_time'] = 0
    
    dt['Длина сообщения'] = dt['Текст сообщения'].apply(lambda x: len(x.split()))
    avg_message_length = dt.groupby('ID урока')['Длина сообщения'].mean()
    try:
        metrics['Длина сообщения'] = avg_message_length.iloc[0]
    except ValueError:
        metrics['Длина сообщения'] = 0
    
    return metrics


def compute_metrics(data, saiga_text):  # Подсчет метрик для одного урока
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

    df_inp['Преобладающий тэг'] = tags_id_to_str.get(most_rag(tags))

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

    df_inp['Преобладающая эмоция'] = tags_id_to_str2.get(find_common_emotion(em_tags))

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
