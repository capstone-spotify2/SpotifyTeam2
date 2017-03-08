import re
import urllib.request
from bs4 import BeautifulSoup
import string
import numpy as np
import pandas as pd
import RequestAgent
import time

##############################
#
# get all artists pages
#
#

import time
import RequestAgent

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
allartists = pd.read_csv('allartists.csv')
#allsongs = []
for i, artist in allartists.ix[6045:6994].iterrows():
    try:
        aname = artist.artist
        alink = artist.artistpage
        url = "http://azlyrics.com/"+alink
        r = RequestAgent.request(url)
        soup = BeautifulSoup(r, 'html.parser')
        soup = str(soup).split("""<script type="text/javascript">""")[1]
        albums = soup.split("""<div class="album">""")[1:]
        albums = list(map(lambda x: """<div class="album">"""+x, albums))
        if len(albums) > 0:
            artistdf = []
            for album in albums:
                asoup = BeautifulSoup(album)
                albuminfo = list(asoup.findAll('div')[0].children)
                if len(albuminfo) == 3:
                    t, name, year = albuminfo
                    name = name.text.replace('"', "")
                    year = year.replace("(", "").replace(")", "").replace(" ", "")
                else:
                    name = np.nan
                    year = np.nan
                albumdf = pd.DataFrame([(song.text, song.get('href')) for song in asoup.findAll('a')])
                albumdf.columns = ['song', 'link']
                albumdf['album'] = name
                albumdf['year'] = year
                artistdf.append(albumdf)
            artistdf = pd.concat(artistdf, ignore_index = True)
            aname = aname.replace("/","_")
            artistdf['artist'] = aname
            artistdf.to_csv('albuminfo/%s.csv'%aname, index = False)
            #allsongs.append(artistdf)
    except RecursionError:
        pass
    time.sleep(15)

#allsongs = pd.concat(allsongs, ignore_index = True)

# a=a.replace('\n','')
# re.findall("<b>(.*?)</b>",a)
# re.findall("</b>(.*?)</div>",a)
# re.findall("<a href=\"(.*?)\".*?>(.*?)<",a)
