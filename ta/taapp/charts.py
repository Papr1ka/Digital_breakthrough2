from django.shortcuts import render
import pandas as pd
from plotly.offline import plot
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from urllib.parse import urlparse

from .models import Lesson, Message
import re
from os import path

from django.conf import settings


DATA = settings.DATA_FOLDER

BAD_WORDS = path.join(DATA, 'Complete Russian badwords dictionary')

with open(BAD_WORDS, "r") as f:
    swears = f.read().splitlines()

exclude = ["fags", "fag", "homo", "around"]
swears = [word.lower() for word in swears[0].split(', ') if word.lower() not in exclude ]

swears = list(set(swears))


def gen_df(lesson_id, bert1=False, bert2=False):
    lesson = Lesson.objects.get(_id=lesson_id)
    messages = lesson.messages()
    data = {
        'ID урока': [],
        'Текст сообщения': [],
        'Дата сообщения': [],
        'Дата старта урока': []
    }
    if bert1:
        data['Тэги на тэги'] = []
    if bert2:
        data['Тэги на эмоции'] = []

    for message in messages:
        data['ID урока'].append(message.lesson._id)
        data['Текст сообщения'].append(message.content)
        data['Дата старта урока'].append(message.lesson.start_time)
        data['Дата сообщения'].append(message.datetime)
        if bert1:
            data['Тэги на тэги'].append(message.tag_bert1)
        if bert2:
            data['Тэги на эмоции'].append(message.tag_bert2)
    df = pd.DataFrame(data=data)
    return df
    
def preprocess_(dt) -> pd.DataFrame:
    dt = dt[(dt['Текст сообщения'].notna()) &
        (dt['ID урока'].notna()) &
        (dt['Дата сообщения'].notna())]
    
    dt['Текст сообщения'] = dt['Текст сообщения'].apply(remove_timestamp)
    dt['ID урока'] = dt['ID урока'].astype(int)
    dt['Дата сообщения'] = pd.to_datetime(dt['Дата сообщения'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    #dt['Дата старта урока'] = pd.to_datetime(dt['Дата старта урока'].str.strip('{}'), format='%Y-%m-%d, %H:%M')
    dt = dt[dt['Дата сообщения'].notna()]
    endtime_table  = dt.groupby(['ID урока'])['Дата сообщения'].max()
    dt = pd.merge(dt, endtime_table, on='ID урока', how='left')
    dt['lesson_time'] = (dt['Дата сообщения_y'] - dt['Дата старта урока']).apply(lambda x: x.components.hours * 60 + x.components.minutes)
    dt = remove_outliers(dt)
    return dt


def plot1():
    data = {'A': np.arange(100), 'B': np.random.randn(100), 'C': np.random.randn(100)}
    df = pd.DataFrame(data=data)
    fig = px.line(df, x="A", y=['B', 'C'], title='Life expectancy in Canada')
    gantt_plot = plot(fig, output_type="div")
    context = {'plot_div': gantt_plot}
    return context

def remove_timestamp(message):
    REGEX = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z?)"
    p = re.compile(REGEX, re.IGNORECASE)
    date = p.findall(message)
    for i in date:
        message = message.replace(i, '').strip()
        if message.endswith(";"):
            message = message[:-1]
    return message


def remove_outliers(dt, time_to_delete=300):
    df_temp = dt.sort_values(['ID урока', 'Дата сообщения_x'])

    df_temp['time_diff'] = df_temp.groupby('ID урока')['Дата сообщения_x'].diff()

    df_temp['time_diff'] = df_temp['time_diff'].dt.total_seconds() / 60  # Преобразование в минуты
    max_time_diff_per_lesson = df_temp.groupby('ID урока')['time_diff'].max()

    lessons_to_keep = max_time_diff_per_lesson[max_time_diff_per_lesson <= time_to_delete].index
    lessons_to_keep = lessons_to_keep.append(max_time_diff_per_lesson[max_time_diff_per_lesson.isnull()].index)

    dt = dt[dt['ID урока'].isin(lessons_to_keep)]
    return dt

def count_messages_per_interval(df, interval='5min'):
    df_copy = df.copy()
    df_copy.loc[:, 'Дата сообщения_x'] = pd.to_datetime(df_copy['Дата сообщения_x'])
    df_copy.loc[:, 'interval_start'] = df_copy['Дата сообщения_x'].dt.floor('-' + interval)

    unique_lesson_ids = df_copy['ID урока'].unique()
    print(unique_lesson_ids)
    print("торт")

    all_intervals = []
    for lesson_id in unique_lesson_ids:
        lesson_df = df_copy[df_copy['ID урока'] == lesson_id]
        min_time = lesson_df['interval_start'].min()
        max_time = lesson_df['interval_start'].max()
        intervals_df = pd.DataFrame({
            'interval_start': pd.date_range(min_time, max_time, freq=interval)
        })
        intervals_df['ID урока'] = lesson_id
        all_intervals.append(intervals_df)

    all_intervals_df = pd.concat(all_intervals)

    count_by_interval = df_copy.groupby(['ID урока', 'interval_start'])['Текст сообщения'].count().reset_index(name='count')
    result_df = all_intervals_df.merge(count_by_interval, on=['ID урока', 'interval_start'], how='left')
    result_df['count'] = result_df['count'].fillna(0)

    return result_df

def find_question_sentences(df):
    def check_question_sentence(message):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', message)

        question_sentences = [sentence.strip() for sentence in sentences if '?' in sentence and 'http' not in sentence]
        if question_sentences:
            return True
        else:
            return False

    df["Question"] = df["Текст сообщения"].apply(check_question_sentence)

    return df


def message_density_and_questions_graph(lesson_id, preprocessed_df=None):
    if preprocessed_df is None:
        lesson_df = gen_df(lesson_id)
        lesson_df = preprocess_(lesson_df)
    else:
        lesson_df = preprocessed_df
    lesson_df = find_question_sentences(lesson_df)
    count_by_interval_lesson = count_messages_per_interval(lesson_df, interval='5min')
    
    if len(count_by_interval_lesson) == 1:
        interval_start = count_by_interval_lesson.iloc[0]['interval_start']
        fig = go.Figure(data=go.Scatter(x=[interval_start], y=[1], mode='markers'))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=count_by_interval_lesson['interval_start'], y=count_by_interval_lesson['count'], mode='lines', name='Сообщения'))
        
        questions_df = lesson_df[lesson_df['Question'] == True]
        if not questions_df.empty: 
            count_by_interval_questions = count_messages_per_interval(questions_df, interval='5min')
        
            merged_data = pd.merge(count_by_interval_lesson, count_by_interval_questions, on='interval_start', how='outer')
            merged_data.fillna(0, inplace=True)
        
            fig.add_trace(go.Scatter(x=merged_data['interval_start'], y=merged_data['count_y'], mode='lines', name='Вопросы', line=dict(dash='dash')))
        
    fig.update_xaxes(title_text="Время", tickformat='%H:%M')
    fig.update_yaxes(title_text="Сообщения")
    fig.update_layout(title=f'График сообщений урока ID{lesson_id}')

    plt = plot(fig, output_type="div")
    context = {'plot_div': plt}
    return context




def contains_url(df):
    def check_url(message):
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"

        matches = re.findall(pattern, message)
        domains = [urlparse(match).netloc for match in matches]
        if len(domains) == 1:
            return ', '.join(domains)
        else:
             return ''

    df["Domain"] = df["Текст сообщения"].apply(check_url)

    return df


def domain_graph(lesson_id, preprocessed_df=None):
    if preprocessed_df is None:
        lesson_df = gen_df(lesson_id)
        lesson_df = preprocess_(lesson_df)
    else:
        lesson_df = preprocessed_df
    lesson_df = contains_url(lesson_df)
    domains = lesson_df[lesson_df["Domain"] != '']['Domain'].value_counts()
    if not(domains.empty):
        fig = px.bar(x=domains.index, y=domains.values, labels={'x': 'Ссылки', 'y': 'Замечено в чате (раз)'}, title=f'Количество различных ссылок на уроке  ID{lesson_id}')
        fig.update_traces(width=0.2)
        fig.update_layout(xaxis_tickangle=-20)
        fig.update_yaxes(tickvals=list(range(max(domains.values) + 1)))
        plt = plot(fig, output_type="div")
        context = {'plot_div': plt}
        return context


def contains_word(df):

    def check_words(message):
        mes =  [i.lower() for i in message.split(' ')]
        for word in swears:
            if word in mes:
                #print(word, mes)
                return True
        return False

    df["Contains word"] = df["Текст сообщения"].apply(check_words)

    return df


def find_swears(lesson_id):
    lesson_df = gen_df(lesson_id)
    lesson_df = preprocess_(lesson_df)
    lesson_df = contains_word(lesson_df)
    swears = lesson_df[lesson_df["Contains word"] == True]
    
    if not swears.empty:
        swear_times = swears[['Дата сообщения_x', 'Текст сообщения']].values.tolist()
        text = f"Плохие слова были обнаружены в уроке ID {lesson_id}. Mоменты времени и тексты сообщений, когда это произошло:"
        for time, swear_text in swear_times:
            text += f"\n{time}: {swear_text}"
        return text
    else:
        return [f"Нет"]


def pipeline_get_met(lid, preprocessed_df=None, plot_metrics=True) -> dict:
    '''
    
    расчет метрик
    с препроцессингом
    return dict
    
    '''
    if preprocessed_df is None:
        dt = gen_df(lid)
        dt = preprocess_(dt)
    else:
        dt = preprocessed_df
    metrics = {
        "avg_message_length": -1,
        "avg_msg_delay": -1,
        "max_msg_delay": -1
    }

    df_merged_1 = dt.sort_values(['ID урока', 'Дата сообщения_x'])
    df_merged_1['time_diff'] = df_merged_1.groupby('ID урока')['Дата сообщения_x'].diff()
    df_merged_1['time_diff'] = df_merged_1['time_diff'].dt.total_seconds() / 60  

    average_time_diff_per_lesson = df_merged_1.groupby('ID урока')['time_diff'].mean()
    max_time_diff_per_lesson = df_merged_1.groupby('ID урока')['time_diff'].max()

    average_time_diff_per_lesson.name = "avg_time"
    max_time_diff_per_lesson.name = "max_time"
    

    metrics['avg_msg_delay'] = dict(average_time_diff_per_lesson)
    metrics['max_msg_delay'] = dict(max_time_diff_per_lesson)
    
    dt['Длина сообщения'] = dt['Текст сообщения'].apply(lambda x: len(x.split()))
    avg_message_length = dt.groupby('ID урока')['Длина сообщения'].mean()
    metrics['avg_message_length'] = dict(avg_message_length)
    if plot_metrics:
        data_to_show = {
            'Metrics': ['Ср. длина сообщения', 'Ср. время между сообщениями', 'Макс. время между сообщениями'],
            'Кол-во минут': [avg_message_length.iloc[0], max_time_diff_per_lesson.iloc[0], average_time_diff_per_lesson.iloc[0]]
        }
        fig = px.bar(data_to_show, x='Metrics', y='Кол-во минут', color='Metrics', title=f'Количественные данные Урока ID{lid}')
        plt = plot(fig, output_type="div")
        context = {'plot_div': plt}
        return context
        
        
    return None

def count_tech_issues_per_interval(df, interval='5min'):
    df_copy = df.copy()
    df_copy.loc[:, 'Дата сообщения_x'] = pd.to_datetime(df_copy['Дата сообщения_x'])
    df_copy.loc[:, 'interval_start'] = df_copy['Дата сообщения_x'].dt.floor('-' + interval)

    unique_lesson_ids = df_copy['ID урока'].unique()

    all_intervals = []
    for lesson_id in unique_lesson_ids:
        lesson_df = df_copy[df_copy['ID урока'] == lesson_id]
        min_time = lesson_df['interval_start'].min()
        max_time = lesson_df['interval_start'].max()
        intervals_df = pd.DataFrame({
            'interval_start': pd.date_range(min_time, max_time, freq=interval)
        })
        intervals_df['ID урока'] = lesson_id
        all_intervals.append(intervals_df)

    all_intervals_df = pd.concat(all_intervals)

    count_by_interval = df_copy.loc[df_copy['Тэги на тэги'] == "технические неполадки"].groupby(['ID урока', 'interval_start']).size().reset_index(name='count')
    result_df = all_intervals_df.merge(count_by_interval, on=['ID урока', 'interval_start'], how='left')
    result_df['count'] = result_df['count'].fillna(0)

    return result_df


def message_density_and_tech_issues_graph(lid, df):

    count_by_interval_lesson = count_tech_issues_per_interval(df, interval='5min')

    if len(count_by_interval_lesson) == 1:
        interval_start = count_by_interval_lesson.iloc[0]['interval_start']
        fig = go.Figure(data=go.Scatter(x=[interval_start], y=[1], mode='markers'))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=count_by_interval_lesson['interval_start'], y=count_by_interval_lesson['count'], mode='lines', name='Технические неполадки'))

    fig.update_xaxes(title_text="Время", tickformat='%H:%M')
    fig.update_yaxes(title_text="Количество сообщений")
    fig.update_layout(title=f'График технических неполадок урока ID{lid}')

    plt = plot(fig, output_type="div")
    context = {'plot_div': plt}
    return context

colors = {"нейтральное": 'lightgray', "негативное": 'red', "позитивное": 'green'}

def pie(lid, lesson_df):
    fig = go.Figure(data=[go.Pie(labels=lesson_df['Тэги на эмоции'].unique(),
                                values=lesson_df['Тэги на эмоции'].value_counts(),
                                marker_colors=[colors[label] for label in lesson_df['Тэги на эмоции'].unique()])])

    fig.update_layout(title=f'Распределение настроений урока ID {lid}')
    plt = plot(fig, output_type="div")
    context = {'plot_div': plt}
    return context
