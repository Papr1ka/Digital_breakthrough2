from django.db import models
from django.urls import reverse

# Create your models here.
class Lesson(models.Model):
    
    _id = models.IntegerField(primary_key=True, unique=True)
    start_time = models.DateTimeField(null=True)
    
    long_description = models.TextField(null=True)
    long_description2 = models.TextField(null=True)
    
    handled = models.BooleanField(default=False)
    handled_bert = models.BooleanField(default=False)
    handled_saiga = models.BooleanField(default=False)
    handled_bert_emo = models.BooleanField(default=False)
    handled_saiga2 = models.BooleanField(default=False)
    
    score = models.FloatField(null=True)
    
    class Meta:
        ordering = ["-handled_saiga", "-_id"]
    
    def __str__(self) -> str:
        return str(self._id)
    
    def messages_count(self):
        return Message.objects.filter(lesson=self).count()
    
    def messages(self):
        return Message.objects.filter(lesson=self)
    
    def get_detail_url(self):
        return reverse("lesson_detail", kwargs={'lid': self._id})

class Message(models.Model):
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    content = models.TextField()
    datetime = models.DateTimeField()
    tag_bert1 = models.CharField(max_length=100, null=True)
    tag_bert2 = models.CharField(max_length=100, null=True)
    # сюда бы ещё можно теги засунуть
    
    class Meta:
        ordering = ["-datetime"]
    
    def __str__(self) -> str:
        return f"{self.lesson.pk}, {self.content}, {self.datetime}"
