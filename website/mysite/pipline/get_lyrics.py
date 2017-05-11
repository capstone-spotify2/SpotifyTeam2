import sys
from optparse import OptionParser
import re
# import urllib.request
from bs4 import BeautifulSoup
import string
import numpy as np
import pandas as pd
import RequestAgent
import time


def LyricsToSentences(lyrics):
    lyrics = strip_non_ascii(lyrics)
    if '\r' in lyrics:
        lines = lyrics.split('\r')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = " "
        lines = ' '.join(lines).replace("\n \n", "\n").split(' \n')
        l = ' '
        for i in lines:
            l+= i
        return l
    elif "\n\n" in lyrics:
        lines = lyrics.split('\n')
        for i, l in enumerate(lines):
            if "<i>" in l:
                lines[i] = " "
        lines = ' '.join(lines).split('\r')
        l = ' '
        for i in lines:
            l+= i
        return l

def get_lyrics(song_title, artist):
    artist = artist.lower()
    song_title = song_title.lower()
    # remove all except alphanumeric characters from artist and song_title
    artist = re.sub('[^A-Za-z0-9]+', "", artist)
    song_title = re.sub('[^A-Za-z0-9]+', "", song_title)
    if artist.startswith("the"):    # remove starting 'the' from artist e.g. the who -> who
        artist = artist[3:]
    url = "http://azlyrics.com/lyrics/"+artist+"/"+song_title+".html"
    print(url)
    
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
        #print (lyrics)
        return lyrics
    except Exception as e:
        lyrics = "Sorry, we cannot find this song lyrics according to the given artist and song name. Please input another song."
        print(lyrics)
        return False

def main(argv=None):
    if argv is None:
        argv = sys.argv

    usage = "Usage: %prog -s song -artist (pass -h for more info)"
    parser = OptionParser(usage)

    parser.add_option("-s", "--song_name", dest="song",
                  help="name of the song")

    parser.add_option("-a", "--artist", dest="artist",
                  help="artist of the song")

    (options, args) = parser.parse_args(argv[1:])

    if options.song is not None and options.artist is not None:
        lyrics = get_lyrics(options.song,options.artist)
        print (lyrics)
    else:
        parser.error("Must pass in song name and artist name")
    return 0

if __name__ == "__main__":
    sys.exit(main())
