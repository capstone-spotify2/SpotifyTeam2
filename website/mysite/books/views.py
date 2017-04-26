from django.shortcuts import render
from books.models import Book
from django.http import HttpResponse
from .forms import UploadFileForm
from django.http import HttpResponseRedirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import cPickle
import os

# from handle_uploaded_file import *
import sys
sys.path.append('../')
from pipline import *

def search_form(request):
    return render(request, 'search_form.html')

def search(request):
    error = False
    if 'q' in request.GET:
        q = request.GET['q']
        if not q:
            error = True
        else:
            recommend = tag_to_song.tag_prediction(raw_data = "./pipline/full_dataset.csv", tag = q)
            return render(request, 'search_form.html',
                          {'recommend': recommend, 'query': q})
    return render(request, 'search_form.html', {'error': error})


def upload_file(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        print(os.getcwd())
        with open('./pipline/model.pickle', 'rb') as f:
            rf = cPickle.load(f)
        tags = model_fitting.model_fit(filename, rf, feature_num = 150)
        return render(request, 'search_form.html', {
            'filename': myfile,
            'tags': tags,

        })
    return render(request, 'search_form.html')


