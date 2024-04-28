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

from collections import Counter
from catboost import CatBoostRegressor


model = CatBoostRegressor()
model.load_model(path, format="cbm")

def get_regressor(path='final'):  # Загрузка модели
    model = CatBoostRegressor()
    model.load_model(path, format="cbm")
    return model


def predict_regressor(model, df_inp):  # Получение оценки
    computed_metrics = compute_metrics(df_inp)
    return model.predict(computed_metrics)


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
    data = contains_questions(data)

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

        most_common_value = counter.most_common(1)[0][0]

        return tags_str_to_id[most_common_value]

    df_inp['Преобладающий тэг'] = most_rag(tags)

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

    df_inp['Преобладающая эмоция'] = find_common_emotion(em_tags)

    rel_data = list(map(check_url, comments))
    rel_data = list(map(classify, rel_data))
    counter = Counter(rel_data)
    df_inp['Тип ссылок'] = counter.most_common(1)[0][0]

    id = data.iloc[0]["ID урока"]
    mes_metrics = pipeline_get_met(data)[0]
    for k, v in mes_metrics.items():
        df_inp[k] = v[id]

    return df_inp
