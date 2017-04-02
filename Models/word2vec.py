import re
import pandas as pd
import nltk.data
from os import listdir
from os.path import isfile, join
from nltk import word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
from gensim.models import Phrases
import logging
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
## to solve the ascii decoding error
import sys
reload(sys)
sys.setdefaultencoding('utf8')

nltk.download('punkt')
tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')

def LyricsToSentences(lyrics):
    """
    This function achieves the following:
    - process raw lyrics to remove tags such as "chorus"
    - break the raw text into sentences
    - in order to avoid extreming short sentences that are
      often out of context, we apply a heuristic where we
      merge any lyric line without a verb with the previous line
    """
    if '\r' in lyrics:
        lines = lyrics.split('\r')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = ""
            if 'VB' not in [t[:2] for w,t in nltk.pos_tag(word_tokenize(l))]:
                lines[i] = l.replace("\n", "")
        lines = ' '.join(lines).replace("\n \n", "\n").split(' \n')
    elif "\n\n" in lyrics:
        lines = lyrics.split('\n')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = ""
            elif 'VB' in [t[:2] for w,t in nltk.pos_tag(word_tokenize(l))]:
                lines[i] = l + "\r"#l.replace("\n", "")
        lines = ' '.join(lines).split('\r')
        
    return lines

def sentence_to_wordlist(sentence, remove_stopwords=False):
    # 1. Remove non-letters
    sentence_text = re.sub(r'[^\w\s]','', sentence)
    # 2. Convert words to lower case and split them
    words = sentence_text.lower().split()
    if remove_stopwords:
        stop = set(stopwords.words('english'))
        filter(lambda x: x not in stop, stop)
    # 3. Return a list of words
    return(words)

def oped_to_sentences(raw_sentences, remove_stopwords=False ):
    """
    Return the list of sentences (each sentence is a list of words, so this returns a list of lists)
    """
    try:
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                sentences.append(sentence_to_wordlist(raw_sentence, remove_stopwords))
        len(sentences)
        return sentences
    except:
        print('nope')


dir = "GetAlbumInfo/lyrics/"
lfiles = [dir+f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith(".txt")]

for i, f in enumerate(lfiles[486+521+4+18+7+10+11+45+293+21+144+165+24847+126+13281:]):
    print f
    file = open(f,'r') 
    lyrics = file.read()
    if (' math' in lyrics.lower())& ('learned' in lyrics.lower()):
        break

sentences = []

for f in lfiles:
    print f
    file = open(f,'r') 
    lyrics = file.read()
    try:
        raw_sentences = LyricsToSentences(lyrics)
        sentences += oped_to_sentences(raw_sentences, remove_stopwords=True)
    except:
        print 'except', f

print("There are " + str(len(sentences)) + " sentences in our corpus of opeds.")
# There are 1731157 sentences in our corpus of opeds.
np.mean([len(s) for s in sentences])
# Out[53]: 7.8848290478564333

for i, s in enumerate(sentences):
    sentences[i] = filter(lambda x: x not in stop, s)

##############################
#
# add literature
#
#####
tokenizer = RegexpTokenizer(r'\w+')
puncs = [',', '.', ']', '[', '!', '?']
lits = []
for lit in nltk.corpus.gutenberg.fileids():
    sents = list(nltk.corpus.gutenberg.sents(lit))
    for i, s in enumerate(sents):
        sents[i] = filter(lambda x: (x not in stop)&(x not in puncs), s)
    lits+=sents

lits = sentences+lits


##############################
#
# building model
#
#####

num_features = 3    # Word vector dimensionality                      
min_word_count = 50   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 5           # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

model = word2vec.Word2Vec(sentences, workers=num_workers,
            size=num_features, min_count = min_word_count, window =
            context, sample = downsampling)

#model.most_similar('happy',  topn=15)
model_name = "model1"
model.save(model_name)


bigramer = gensim.models.Phrases(sentences)

bmodel = Word2Vec(bigramer[sentences], workers=num_workers,
            size=num_features, min_count = min_word_count, 
            window = context, sample = downsampling)

bmodel.save('bmodel3d')


##############################
#
# testing happy vs. sad
#
#####

happy = pd.read_csv('happy_counts.csv', index_col = 0)
sad = pd.read_csv('sad_counts.csv', index_col = 0)
np.sum(happy.mxm_id.isin(sad.mxm_id))
# Out[105]: 346
shareid = happy.mxm_id[happy.mxm_id.isin(sad.mxm_id)]
happy = happy[~happy.mxm_id.isin(shareid)]
sad = sad[~sad.mxm_id.isin(shareid)]

happy = happy.set_index('mxm_id')
sad = sad.set_index('mxm_id')
songw = {}
for i, song in happy.iterrows():
    songw[i] = filter(lambda x: x not in stop, list(song[song!=0].index[1:]))
for i, song in sad.iterrows():
    songw[i] = filter(lambda x: x not in stop, list(song[song!=0].index[1:]))


vocab = list(model.vocab.keys())
modeldf = pd.DataFrame(model.syn0)
modeldf.index = vocab

def findDistance(wlist, w, modeldf):
    wlistpos = modeldf.ix[wlist].sum()
    return np.linalg.norm(wlistpos - modeldf.ix[w])

output = [pd.Series([findDistance(words, 'happy', modeldf), findDistance(words, 'sad', modeldf)]) for k, words in songw.iteritems()]

output = pd.concat(output, axis = 1).T
output['mxm_id'] = songw.keys()
output['happy'] = (output[0] < output[1])
output['happy'] = (output[0] - output[1])/abs(output[0]) < -0.1
output['sad'] = (output[0] - output[1])/abs(output[1]) > 0.1
