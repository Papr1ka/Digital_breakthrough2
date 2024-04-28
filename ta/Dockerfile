FROM python:3.10
ENV PYTHONUNBUFFERED=1
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN python manage.py makemigrations && python manage.py runserver && python manage.py convert /usr/src/data/saiga_texts.json && python manage.py convert2 /usr/src/data overall_swears.json
CMD celery -A taapp.celery worker -P solo -l info
