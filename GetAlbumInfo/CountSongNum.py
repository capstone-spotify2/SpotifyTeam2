import pandas as pd
from os import listdir
from os.path import isfile, join

####################################
#
# count songs within a letter range, and
# after (including) an optional start artist
#
# Lower case for start_letter and end_letter

def count_songs(start_letter, end_letter, start_artist=None):
    afiles = [f for f in listdir("albuminfo") if isfile(join('albuminfo', f)) and f.endswith(".csv")
              and start_letter <= f[0].lower() <= end_letter]
    if start_artist:
        afiles = afiles[afiles.index('%s.csv' % start_artist):]
    count = 0
    for f in afiles:
        ainfo = pd.read_csv('albuminfo/'+f)
        for i, song in ainfo.iterrows():
            if isinstance(song.song, str) and isinstance(song.artist, str):
                count += 1
    print("Num of songs from '%s' to '%s' start with artist '%s': %d" % (start_letter, end_letter, start_artist, count))
    # 351393 in total

count_songs('m', 'm')
