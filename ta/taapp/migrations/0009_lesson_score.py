# Generated by Django 5.0.4 on 2024-04-28 00:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('taapp', '0008_lesson_handled_saiga2_lesson_long_description2'),
    ]

    operations = [
        migrations.AddField(
            model_name='lesson',
            name='score',
            field=models.FloatField(null=True),
        ),
    ]
