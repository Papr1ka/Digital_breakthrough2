from celery import shared_task
from .models import Message, Lesson
import pandas as pd
import re
from django.core.files.storage import FileSystemStorage
import logging

logging.error(f"{__name__} {__loader__}")

from .charts import gen_df, preprocess_
from .metrics import compute_metrics

REGEX = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z?)"
p = re.compile(REGEX, re.IGNORECASE)

@shared_task
def handle_file(file):
    file = FileSystemStorage(location='tmp').path(file)
    df = pd.read_excel(file, engine="calamine")
    
    uniq_ids = set()
    uniq_rows = []
    
    for row in df.values:
        if not pd.isna(row[0]) and not pd.isna(row[1]):
            if not row[0] in uniq_ids:
                uniq_ids.add(row[0])
                uniq_rows.append((int(row[0]), row[1]))
    
    
    new_lessons = {ind: Lesson(_id=ind, start_time=t) for ind, t in uniq_rows}
    Lesson.objects.bulk_create(new_lessons.values(), ignore_conflicts=True)
    
    def remove_timestamp(message):
        if pd.isna(message):
            return ""
        date = p.findall(message)
        for i in date:
            message = message.replace(i, '').strip()
            if message.endswith(";"):
                message = message[:-1]
        return message
    
    messages = [Message(
        lesson=new_lessons[int(i[0])],
        content=remove_timestamp(i[3]),
        datetime=i[4]
        ) for i in df.values if not (pd.isna(i[0]) or pd.isna(i[4]))]
    Message.objects.bulk_create(messages)
    return True

imported = False
def make_import():
    global predict_text, generate_resume_comments, predict_text2, predict_regressor, generate_final_grade
    from .predict import predict_text, generate_resume_comments, predict_text2, predict_regressor, generate_final_grade

@shared_task
def predict_text_prod(*args, **kwargs):
    if not imported:
        make_import()
    return predict_text(*args, **kwargs)

@shared_task
def predict_text_prod2(*args, **kwargs):
    if not imported:
        make_import()
    return predict_text2(*args, **kwargs)

@shared_task
def saiga(lid):
    if not imported:
        make_import()
    lesson = Lesson.objects.get(_id=lid)
    messages = lesson.messages()
    data = "\n".join([message.content for message in messages])
    descr = generate_resume_comments(data)
    lesson.long_description = descr
    lesson.handled_saiga = True
    lesson.save(force_update=True)
    return True

@shared_task
def regressor(lid):
    if not imported:
        make_import()
    lesson = Lesson.objects.get(_id=lid)
    df = gen_df(lid, bert1=True, bert2=True)
    df = preprocess_(df)
    score = predict_regressor(df, lesson.long_description)
    lesson.score = score
    lesson.save()
    return True

@shared_task
def saiga2(lid):
    if not imported:
        make_import()
    lesson = Lesson.objects.get(_id=lid)
    df = gen_df(lid, bert1=True, bert2=True)
    df = preprocess_(df)
    metrics = compute_metrics(df, lesson.long_description)
    out = generate_final_grade(metrics)
    if out.startswith("assistant"):
        out = out.replace("assistant", "").strip()
    lesson.long_description2 = out
    lesson.save()
    return True
