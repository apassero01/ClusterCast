from django.shortcuts import render
import os

def index(request, *args, **kwargs):
    # print files in templates folder
    # import os
    for root, dirs, files in os.walk('frontend/templates'):
        for file in files:
            print(file)
    return render(request, 'frontend/index.html')
