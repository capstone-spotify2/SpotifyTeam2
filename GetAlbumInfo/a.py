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

def get_lyrics(url):
    try:
        #content = urllib.request.urlopen(url).read()
        r = RequestAgent.request(url)
        soup = BeautifulSoup(r, 'html.parser')
        lyrics = str(soup)
        # lyrics lies between up_partition and down_partition
        up_partition = '<!-- Usage of azlyrics.com content by any third-party lyrics provider is prohibited by our licensing agreement. Sorry about that. -->'
        down_partition = '<!-- MxM banner -->'
        lyrics = lyrics.split(up_partition)[1]
        lyrics = lyrics.split(down_partition)[0]
        lyrics = lyrics.replace('<br>','').replace('</br>','').replace('</div>','').strip()
        return lyrics
    except Exception as e:
        raise Exception("Exception occurred when getting url %s. %s" % (url, str(e)))

def crawl_one_artist_from_song(afile, stopped_song=None):
    ainfo = pd.read_csv('albuminfo/' + afile)
    start_crawling = False if stopped_song else True
    for i, song in ainfo.iterrows():
        if start_crawling:
            if isinstance(song.song, str) and isinstance(song.artist, str) and isinstance(song.link, str):
                url = song.link
                if url.startswith('../'):
                    url = "http://www.azlyrics.com/" + url[3:]
                elif url.startswith('http'):
                    pass
                else:
                    print("Not able to process url: %s" % url)
                    continue
                try:
                    lyrics = get_lyrics(url)
                except Exception as e:
                    print("Error: %s_%s.txt, %s. %s" % (song.artist.replace("/", "_"), song.song.replace("/", "_"), url, e))
                else:
                    if lyrics:
                        with open("lyrics/%s_%s.txt"%(song.artist.replace("/", "_"), song.song.replace("/", "_")), "w") as f:
                            f.write(lyrics)
                time.sleep(10)
        elif isinstance(song.song, str) and song.song == stopped_song:
            start_crawling = True


def crawl_lyrics(stopped_artist, stopped_song, start_letter, end_letter):
    crawl_one_artist_from_song('%s.csv' % stopped_artist, stopped_song)

    afiles = [f for f in listdir("albuminfo") if isfile(join('albuminfo', f)) and f.endswith(".csv")
              and start_letter <= f[0].lower() <= end_letter]
    afiles = afiles[afiles.index('%s.csv' % stopped_artist)+1:]

    for afile in afiles:
        crawl_one_artist_from_song(afile)


stopped_artist = 'MEREDITH ANDREWS'
stopped_song = 'Christ Is Enough'
start_letter = 'm'
end_letter = 'm'
crawl_lyrics(stopped_artist, stopped_song, start_letter, end_letter)