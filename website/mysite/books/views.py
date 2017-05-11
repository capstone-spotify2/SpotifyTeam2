from django.shortcuts import render
from books.models import Book
from django.http import HttpResponse
from .forms import UploadFileForm
from django.http import HttpResponseRedirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import pandas as pd
import cPickle
import os

# from handle_uploaded_file import *
import sys
sys.path.append('../')
from pipline import *

def index(request):
    return render(request, 'index.html')

def predict(request):
    print "lalala"
    if request.method == 'GET':
        song = request.GET['song']
        artist = request.GET['artist']
        lyrics = get_lyrics.get_lyrics(song, artist)
        print lyrics
        if type(lyrics)==bool:
            error = True
            flag = True
            return render(request, 'index.html', {
            	'song': song,
            	'artist': artist,
                'error': error,
                'flag': True
                })
        else:
            filename = song+'_'+artist+'.txt'
            f = open(filename, 'w+')
            f.write(lyrics)
            with open('./pipline/model.pickle', 'rb') as f:
                rf = cPickle.load(f)
            tags = model_fitting.model_fit(filename, rf, feature_num = 150)
            return render(request, 'index.html', {
                'song': song,
                'artist': artist,
                'tags': tags,
                'lyric': lyrics,
                'flag': True
            })

def recommend(request):
    error = False
    if 'q' in request.GET:
        q = request.GET['q']
        if not q:
            error = True
        else:
            recommend = tag_to_song.tag_prediction(raw_data = "./pipline/playlist.csv",tag = q)
            return render(request, 'index.html',
                          {'recommend': recommend, 'query': q})
    return render(request, 'index.html', {'error': error})

