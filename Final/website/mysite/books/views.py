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
from keras.models import load_model
from gensim.models.keyedvectors import KeyedVectors
# from jsonify.decorators import ajax_request
# import jason
from django.http import JsonResponse

# from handle_uploaded_file import *
import sys
sys.path.append('../')
from backend import *

######### preload model ##############
# print "loading model...."
# model = load_model('./backend/multiclass_model.h5')
# print "model loaded.."
# ######### preload model ##############


# ######### preload word2vex ###########
print "loading word2vec...."
word2vec = KeyedVectors.load_word2vec_format('../../pipeline/1-LSTM_model_for_tag_prediction/GoogleNews-vectors-negative300.bin', binary=True)
print "word2vec loaded...."
######### preload word2vec ###########

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
            string_song_info = ''+song+' - '+artist
            warning = 'Sorry, we cannot find the lyrics according to the given artist and song name. Please input another song.'

            return JsonResponse({
                'song_info': None,
                'tags': None,
                'lyric': None,
                'warning': warning,
                })

        else:
            ############ <old version> ################################
            filename = song+'_'+artist+'.txt'
            f = open(filename, 'w+')
            f.write(lyrics)
            with open('./backend/model.pickle', 'rb') as f:
                rf = cPickle.load(f)
            print filename
            tags = model_fitting.model_fit(filename, rf, feature_num = 150)
            ############ <old version> ################################

            ############# comment this block <new version> ############
            # tags, prob = model_fitting_0.model_fit(lyrics, model)
            print tags
            ############ comment this block <new version> #############

            string_song_info = ''+song+' - '+artist
            string_response = 'The top 3 tags are: '+tags[0]+', '+tags[1]+', '+tags[2]

            print string_song_info
            print string_response

            return JsonResponse({
                'song_info': string_song_info,
                'tags': string_response,
                'lyric': lyrics,
                'warning': None,
                })



def recommend(request):
    error = False
    if 'tag_input' in request.GET:
        tag = request.GET['tag_input']
        print tag
        recommend = tag_to_song_0.tag_prediction(word2vec, raw_data = "./backend/playlist.csv", tag = tag)
        print recommend
        if recommend == None:
            warning = 'Sorry! Do not have songs for this tag.'
            print warning
            return JsonResponse({
                'rmd_warning': warning,
                'rmd_name_0': None,
                'rmd_name_1': None,
                'rmd_name_2': None,
                'rmd_name_3': None,
                'rmd_name_4': None,
                'rmd_name_5': None,
                'rmd_name_6': None,
                'rmd_name_7': None,
                'rmd_name_8': None,
                'rmd_name_9': None,
                'rmd_link_0': None,
                'rmd_link_1': None,
                'rmd_link_2': None,
                'rmd_link_3': None,
                'rmd_link_4': None,
                'rmd_link_5': None,
                'rmd_link_6': None,
                'rmd_link_7': None,
                'rmd_link_8': None,
                'rmd_link_9': None,
                })
        else:
            name_list = []
            for i in range(10):
                name_str = 'Song: ' + recommend['name'][i] + ', Artist: ' + recommend['artist'][i]
                name_list.append(name_str)
            print name_list
            
            return JsonResponse({
                'rmd_warning': None,
                'rmd_name_0': name_list[0],
                'rmd_name_1': name_list[1],
                'rmd_name_2': name_list[2],
                'rmd_name_3': name_list[3],
                'rmd_name_4': name_list[4],
                'rmd_name_5': name_list[5],
                'rmd_name_6': name_list[6],
                'rmd_name_7': name_list[7],
                'rmd_name_8': name_list[8],
                'rmd_name_9': name_list[9],
                'rmd_link_0': recommend['link'][0],
                'rmd_link_1': recommend['link'][1],
                'rmd_link_2': recommend['link'][2],
                'rmd_link_3': recommend['link'][3],
                'rmd_link_4': recommend['link'][4],
                'rmd_link_5': recommend['link'][5],
                'rmd_link_6': recommend['link'][6],
                'rmd_link_7': recommend['link'][7],
                'rmd_link_8': recommend['link'][8],
                'rmd_link_9': recommend['link'][9],
                })


        
