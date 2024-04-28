from django.urls import path
from .views import (
    HomeView,
    LessonDetailView,
    LoadDatasetView,
    get_task_status_view,
    ready_status
)

urlpatterns = [
    path('', HomeView.as_view(), name="home"),
    path('lesson/<int:lid>/', LessonDetailView.as_view(), name="lesson_detail"),
    path('load/', LoadDatasetView.as_view(), name="load"),
    path('task_status/<str:task_id>', get_task_status_view, name="task_status"),
    path('is_lesson_ready/<int:lid>/', ready_status, name="lesson_ready")
]
