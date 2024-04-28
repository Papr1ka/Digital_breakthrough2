from django import forms


class DatasetForm(forms.Form):
    dataset = forms.FileField(required=True, widget=forms.FileInput(attrs={"class": "form-control form-control-lg", "accept": '.xlsx'}))
