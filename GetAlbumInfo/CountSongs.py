import re
import urllib.request
from bs4 import BeautifulSoup
import string
import numpy as np
import pandas as pd
import RequestAgent
import time
from os import listdir
from os.path import isfile, join

##############################
#
# get all artists pages
#
#

# allartists = []
# for l in string.ascii_lowercase:
#     url = "http://azlyrics.com/%s.html"%l
#     r = RequestAgent.request(url)
#     soup = BeautifulSoup(r, 'html.parser')
#     soup = BeautifulSoup(str(soup).split("container main-page")[1])
#     artistdf = pd.DataFrame([(artist.text, artist.get('href')) for artist in soup.findAll('a')])
#     artistdf.columns = ['artist', 'artistpage']
#     allartists.append(artistdf)
#     # links += [link.get('href') for link in soup.find_all('a') if 'http' not in link.get('href')]
#     time.sleep(15)

# allartists = pd.concat(allartists, ignore_index = True)
# allartists.to_csv('allartists.csv', index = False)


####################################
#
# get all album and songs info
#
#
# allartists = pd.read_csv('allartists.csv')
# #allsongs = []
# for i, artist in allartists.ix[2497:].iterrows():
#     try:
#         aname = artist.artistpage
#         alink = artist.artistpage
#         url = "http://azlyrics.com/"+alink
#         r = RequestAgent.request(url)
#         soup = BeautifulSoup(r, 'html.parser')
#         soup = str(soup).split("""<script type="text/javascript">""")[1]
#         albums = soup.split("""<div class="album">""")[1:]
#         albums = list(map(lambda x: """<div class="album">"""+x, albums))
#         if len(albums) > 0:
#             artistdf = []
#             for album in albums:
#                 asoup = BeautifulSoup(album)
#                 albuminfo = list(asoup.findAll('div')[0].children)
#                 if len(albuminfo) == 3:
#                     t, name, year = albuminfo
#                     name = name.text.replace('"', "")
#                     year = year.replace("(", "").replace(")", "").replace(" ", "")
#                 else:
#                     name = np.nan
#                     year = np.nan
#                 albumdf = pd.DataFrame([(song.text, song.get('href')) for song in asoup.findAll('a')])
#                 albumdf.columns = ['song', 'link']
#                 albumdf['album'] = name
#                 albumdf['year'] = year
#                 artistdf.append(albumdf)
#             artistdf = pd.concat(artistdf, ignore_index = True)
#             aname = aname.replace("/","_")
#             artistdf['artist'] = aname
#             artistdf.to_csv('albuminfo/%s.csv'%aname, index = False)
#             #allsongs.append(artistdf)
#     except RecursionError:
#         pass
#     time.sleep(15)

#allsongs = pd.concat(allsongs, ignore_index = True)

# a=a.replace('\n','')
# re.findall("<b>(.*?)</b>",a)
# re.findall("</b>(.*?)</div>",a)
# re.findall("<a href=\"(.*?)\".*?>(.*?)<",a)


####################################
#
# get lyrics
#
#

#allartists = pd.read_csv('allartists.csv', index_col = 0)

# def get_lyrics(url):
#     try:
#         #content = urllib.request.urlopen(url).read()
#         r = RequestAgent.request(url)
#         soup = BeautifulSoup(r, 'html.parser')
#         lyrics = str(soup)
#         # lyrics lies between up_partition and down_partition
#         up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
#         down_partition = '<!-- MxM banner -->'
#         lyrics = lyrics.split(up_partition)[1]
#         lyrics = lyrics.split(down_partition)[0]
#         lyrics = lyrics.replace('<br>','').replace('</br>','').replace('</div>','').strip()
#         return lyrics
#     except Exception as e:
#         return "Exception occurred \n" +str(e)


afiles = [f for f in listdir("albuminfo") if isfile(join('albuminfo', f)) and f.endswith(".csv")]
count = 0
for f in afiles:
    try:
        ainfo = pd.read_csv('albuminfo/'+f)
        for i, song in ainfo.iterrows():
            if isinstance(song.song, str) and isinstance(song.artist, str):
                count += 1
    except Exception as e:
        print("%s is bad file: %s" % (f, e))

print(count)
# all songs = 351393

            # lyrics = get_lyrics(url)
            # with open("lyrics/%s_%s.txt"%(song.artist.replace("/", "_"), song.song.replace("/", "_")), "w") as text_file:
                # text_file.write(lyrics)

        # time.sleep(15)




# brackets_regex = re.compile('\[.*?\]')
# lyrics = brackets_regex.sub("", lyrics)

# # Remove apostrophes
# apostrophe_regex = re.compile("[']")
# lyrics = apostrophe_regex.sub("", lyrics)

# # Remove punctuation
# punctuation_regex = re.compile('[^0-9a-zA-Z ]+')
# punctuation_regex.sub(" ", lyrics)

# # Remove double spaces
# double_space_regex = re.compile('\s+')
# double_space_regex.sub(" ", lyrics)

