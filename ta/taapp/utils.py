from django.core.management.base import BaseCommand, CommandError
from taapp.models import Lesson, Message
from os import path
import pandas as pd
from pathlib import Path
import re


def handle_file(file):
    df = pd.read_excel(file, engine="calamine")
    
    new_lessons = df['ID урока'].unique()
    new_lessons = {i: Lesson(_id=int(i)) for i in new_lessons if not pd.isna(i)}
    Lesson.objects.bulk_create(new_lessons.values(), ignore_conflicts=True)
    
    REGEX = r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z?)"

    p = re.compile(REGEX, re.IGNORECASE)
    
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
