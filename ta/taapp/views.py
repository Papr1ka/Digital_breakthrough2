from django.shortcuts import render
from django.views import View
from django.views.generic import ListView
from django.conf import settings
from .models import Lesson, Message
import pandas as pd
from random import randint
from dataclasses import dataclass
from .charts import plot1, gen_df, message_density_and_questions_graph, domain_graph, find_swears, pipeline_get_met, gen_df, preprocess_, message_density_and_tech_issues_graph
from django.core.paginator import Paginator
from .forms import DatasetForm
from .tasks import handle_file
from os import path
from django.core.files.storage import FileSystemStorage
from celery.result import AsyncResult
from django.http import JsonResponse
from django.urls import reverse
from .tasks import predict_text_prod, saiga, predict_text_prod2, regressor, saiga2
from .celery import app
from typing import Any
import re
from .metrics import compute_metrics


TEXT = 'Lorem Ipsum - это текст-"рыба", часто используемый в печати и вэб-дизайне. Lorem Ipsum является стандартной "рыбой" для текстов на латинице с начала XVI века. В то время некий безымянный печатник создал большую коллекцию размеров и форм шрифтов, используя Lorem Ipsum для распечатки образцов. Lorem Ipsum не только успешно пережил без заметных изменений пять веков, но и перешагнул в электронный дизайн. Его популяризации в новое время послужили публикация листов Letraset с образцами Lorem Ipsum в 60-х годах и, в более недавнее время, программы электронной вёрстки типа Aldus PageMaker, в шаблонах которых используется Lorem Ipsum.'

TEXT = "Оценка понимания: средне; Список проблем: отсутствие конкретики в объяснении, сложные термины, недостаток практических примеров."

processing = {}

@dataclass
class A:
    name: str
    values: list

@dataclass
class Plot:
    plot: Any
    plot_id: str

def parse_description(text):
    out = []
    
    blocks = text.strip().split(";")
    
    for block in blocks:
        problem, args = block.split(":")
        args = args.split(",")
        out.append(A(problem, args))
    return out

def process(obj):
    return {
        'id': obj._id,
        'messages': obj.messages_count(),
        'description': parse_saiga_text(obj.long_description) if obj.long_description else None,
        'description2': obj.long_description2 if obj.long_description2 else None,
        'url': obj.get_detail_url(),
        'score': obj.score if obj.score else None
    }


@dataclass
class SaigaView:
    distructing: str
    termins: str
    undertanding: str
    discussions: str
    problems: str

def parse_saiga_text(text):  # Парсинг текста. text - текст, генерируемый Сайгой

    def get_part(pattern, id):
        res = re.findall(pattern, text)
        if res:
            return '.'.join(res)
        else:
            return 'нет'
    
    def get_part_list(pattern, id):
        res = re.findall(pattern, text)
        if res:
            return res
        else:
            return ['нет']
    
    return SaigaView(
        get_part(r'Ученики отвлекались - (\w+)', text),
        get_part(r'\d+\.\s(.+?) \(\d+\)', text),
        get_part(r'Оценка понимания - (.+)', text),
        get_part(r'Главная тема - (\w+)', text),
        get_part_list(r'\d+\.\s(.*?\.)(?=\n\d+\.|\Z)', text)
    )


class HomeView(View):
    template_name = "taapp/home.html"
    paginate_by = 15
    queryset = Lesson.objects.all()

    def get(self, request, *args, **kwargs):
        lid = request.GET.get("lid")
        if lid:
            self.queryset = Lesson.objects.filter(_id=lid)
        
        paginator = Paginator(self.queryset, self.paginate_by)
        page_number = request.GET.get("page")
        page_obj = paginator.get_page(page_number)
        
        page_obj.object_list = [process(obj) for obj in page_obj.object_list]
        
        ctx = {
            'page_obj': page_obj
        }
        
        return render(request, self.template_name, context=ctx)


class LessonDetailView(View):
    template_name = "taapp/lesson_detail.html"

    def get(self, request, lid):
        print(lid)
        obj = Lesson.objects.get(_id=lid)
        plots = [plot1(), plot1()]
        
        ctx = {
            'object': obj,
        }
        
        if obj.long_description:
            ctx['description'] = parse_saiga_text(obj.long_description)
        if obj.long_description2:
            ctx['description2'] = obj.long_description2
        if obj.score:
            ctx['score'] = obj.score
            
        messages = obj.messages()
        messages = [i.content for i in messages]
        print(obj.handled)
        print(processing)
        gen_df(lid)
        plots = []
        
        
        df = gen_df(lid)
        df = preprocess_(df)
        
        plots.append(Plot(
            message_density_and_questions_graph(lid, preprocessed_df=df),
            'messages_chart'
        ))
        
        g = domain_graph(lid, preprocessed_df=df)
        if g:
            plots.append(Plot(
                g,
                'links_chart'
            ))
        g = pipeline_get_met(lid, preprocessed_df=df)
        if g:
            plots.append(Plot(
                g,
                'additional'
            ))
            
        swears = find_swears(lid)
        ctx.update(swears=swears)
        
        ctx.update(charts=plots)
        
        
        if obj.handled_bert and obj.handled_bert_emo and obj.handled_saiga:
            # Разкоментить, когда регрессор будет готов
            df = gen_df(lid, bert1=True, bert2=True)
            df = preprocess_(df)
            computed_metrics = compute_metrics(df, obj.long_description).to_frame()
            computed_metrics = [{'name': name, 'value': value[0]} for name, value in zip(computed_metrics.index, computed_metrics.values)]
            
            ctx.update(metrics=computed_metrics)
        
        if obj.handled:
            print("handled")
        
        if obj.handled_bert:
            df = gen_df(lid, bert1=True, bert2=True)
            df = preprocess_(df)
            g = message_density_and_tech_issues_graph(lid, df)
            plots.append(Plot(
                g,
                'tech'
            ))

        if not obj.handled and processing.get(lid) is None:
            print("starting processing...")
            task_ids = []
            if not obj.handled_bert:
                r = predict_text_prod.delay(lid)
                task_ids.append(r)
            if not obj.handled_saiga:
                r2 = saiga.delay(lid)
                task_ids.append(r2)
            if not obj.handled_bert_emo:
                r3 = predict_text_prod2.delay(lid)
                task_ids.append(r3)
            if obj.handled_bert and obj.handled_bert_emo and obj.handled_saiga:
                # Разкоментить, когда регрессор будет готов
                
                if not obj.handled_saiga2:
                    r4 = saiga2.delay(lid)
                    task_ids.append(r4)
                
                #pass
            if task_ids:
                processing[lid] = task_ids
            ctx.update(ready_url=reverse("lesson_ready", kwargs={'lid': lid}))
            ctx.update(script_required=True)
            
            
        elif not obj.handled and processing.get(lid) is not None:
            results = [AsyncResult(i).ready() for i in processing.get(lid)]
            if all(results):
                processing.pop(lid)
                if obj.handled_bert_emo and obj.handled_bert and obj.handled_saiga and obj.score:
                    obj.handled = True
                    obj.save(force_update=True)
                print("processed, deleting and saving")
            elif any(results):
                for i, is_ready in zip(processing.get(lid), results):
                    if is_ready:
                        processing[lid].remove(i)
                ctx.update(script_required=True)
                ctx.update(ready_url=reverse("lesson_ready", kwargs={'lid': lid}))
            else:
                print(f"{len(processing.get(lid))} tasks yet")
                ctx.update(script_required=True)
                ctx.update(ready_url=reverse("lesson_ready", kwargs={'lid': lid}))

        
        return render(request, self.template_name, context=ctx)

class LoadDatasetView(View):
    template_name = "taapp/load.html"
    
    def get(self, request):
        form: DatasetForm = DatasetForm(request.POST, request.FILES)
        data = dict(form=form)

        return render(request, self.template_name, context=data)

    def post(self, request):
        form: DatasetForm = DatasetForm(request.POST, request.FILES)
        data = dict(form=form)
        
        if form.is_valid():
            f = request.FILES['dataset']
            FileSystemStorage(location='tmp').save(f.name, f)
            
            r = handle_file.delay(f.name)
            data.update(message="Датасет поставлен в обработку")
            data.update(task_id=r)
            data.update(task_url=reverse("task_status", kwargs={'task_id': r}))
            data.update(script_required=True)

        return render(request, self.template_name, context=data)

def get_task_status_view(request, task_id):
    task = AsyncResult(task_id)
    print(task)
    print(task.ready())
    return JsonResponse({'ready': task.ready()})

def gen_plot1(request, lid):
    obj = Lesson.objects.get(_id=lid)
    return obj.long_description

def ready_status(request, lid):
    obj = Lesson.objects.get(_id=lid)
    if obj.handled:
        return JsonResponse({'response': 2})
    l = processing.get(lid)
    
    if l is not None:
        print(len(l))
        results = [AsyncResult(i).ready() for i in l]
        if all(results):
            processing.pop(lid)
            if obj.handled_bert_emo and obj.handled_bert and obj.handled_saiga and obj.score:
                obj.handled = True
                obj.save(force_update=True)
            print("processed, deleting and saving from ready status")
            return JsonResponse({'response': 2})
        elif any(results):
            print(results)
            for i, is_ready in zip(processing.get(lid), results):
                if is_ready:
                    processing[lid].remove(i)
                    print(f"task {i} removed...")
            return JsonResponse({'response': 1})
        else:
            return JsonResponse({'response': 0})
    return JsonResponse({"response": -1})
