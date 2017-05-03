import os
import requests
import random

USER_AGENTS_FILE = "user_agents.txt"

def LoadUserAgents(uafile):
    """
    uafile : string
        path to text file of user agents, one per line
    """
    uas = []
    with open(uafile, 'rb') as uaf:
        for ua in uaf.readlines():
            if ua:
                uas.append(ua.strip()[1:-1-1])
    random.shuffle(uas)
    return uas


def request(url, rdagent = False):
    if rdagent:
        # load the user agents, in random order
        uas = LoadUserAgents(uafile=USER_AGENTS_FILE)
        ua = random.choice(uas)  # select a random user agent
    else:
        ua = "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)"
    headers = {'Accept': 'application/json, text/javascript, */*; q=0.01', 
               'Accept-Encoding':'gzip, deflate',
               'Accept-Language':'en-US,en;q=0.8,zh-CN;q=0.6,zh-TW;q=0.4',
               'User-Agent':ua}
    r = requests.get(url, headers = headers)

    return r.text

