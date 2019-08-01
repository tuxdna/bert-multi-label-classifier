import requests
import selenium
import bs4
import urllib
from bs4 import BeautifulSoup as BS
import os
import sys


data_path = None

if len(sys.argv) >= 2:
    data_path = sys.argv[1]

if data_path is None or not os.path.exists(data_path):
    print("Please provide a valid data path. Exiting now ...")
    sys.exit(-1)

movie_script_path = os.path.join(data_path, "valkaama-script.txt")

if not os.path.exists(movie_script_path):
    scenes = []
    for i in range(1, 57):
        url = "http://www.valkaama.com/index.php?page=script&l=en&scene=%d" % i
        html = urllib.request.urlopen(url).read()
        soup = BS(html, features="lxml")
        css_selector = '#myMainBoxContent > div > div:nth-child(12)'
        print(url)
        text = ""
        for e in soup.select(css_selector):
            text += e.text.strip()
        # print(text)
        scenes.append(text)

    with open(movie_script_path, "w") as f:
        f.write("\n\n".join(scenes))


from spacy.lang.en import English
with open(movie_script_path) as f:
    movie_script = f.read()
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    doc = nlp(movie_script)
    for s in doc.sents:
        print("SENT:", str(s))
