# Generated by Django 5.0.4 on 2024-04-27 21:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('taapp', '0005_lesson_handled_bert_lesson_handled_bert_emo_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='message',
            old_name='tag',
            new_name='tag_bert1',
        ),
        migrations.AddField(
            model_name='message',
            name='tag_bert2',
            field=models.CharField(max_length=100, null=True),
        ),
    ]
