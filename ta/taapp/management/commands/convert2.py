from django.core.management.base import BaseCommand, CommandError
from taapp.models import Lesson, Message
from os import path
import pandas as pd
from pathlib import Path
import re
import json

class Command(BaseCommand):
    help = "Loads descriptions from json"

    def add_arguments(self, parser):
        parser.add_argument("file", type=Path)

    def handle(self, *args, **options):
        
        file = options.get('file')
        if not file:
            self.stdout.write(
                self.style.ERROR('Please, specify filename by file=...')
            )
        else:
            file = path.abspath(path.relpath(file))
            
            self.stdout.write(
                self.style.SUCCESS('Reading file...')
            )
            with open(file) as f:
                js = json.loads(f.read())

            for lid, descr in js.items():
                lid = int(float(lid))
                lesson = Lesson.objects.get(_id=lid)
                lesson.long_description2 = descr
                lesson.handled_saiga2 = True
                lesson.save()
            
            self.stdout.write(
                self.style.SUCCESS('SUCCESS!')
            )
