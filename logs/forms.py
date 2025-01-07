from django import forms

class LogFileUploadForm(forms.Form):
    file = forms.FileField(
        label='Select a log file',
        help_text='Supported formats: .log, .txt',
        widget=forms.ClearableFileInput(attrs={'accept': '.log,.txt'}),
    )

