# collecting relation surfaces from OpenIE
# http://openie.allenai.org/search?arg1=book&rel=&arg2=entertainment&corpora=
import requests
import re
from multiprocessing.dummy import Pool
import time
surface_dict = None

def get_relation(subj, obj):
    global surface_dict
    url = "http://openie.allenai.org/search?arg1=%s&rel=&arg2=%s&corpora=" % (subj, obj)
    page = requests.get(url)
    # print(url)
    if page.status_code != 200:
        print(url.encode("utf-8"))
        print(page.status_code)
        time.sleep(1)
        return

    content = page.text
    surfaces = []
    if len(content) >=1000:
        pat = re.compile('<span class="title-string">*(.*)\s*<\/span>\s*\((\d)\)<\/a><\/li>')
        for r in pat.findall(content):
            # print(r)  # r[0] is the relation surface, r[1] is the frequency
            surfaces.append((r[0], int(r[1])))
    for s, f in surfaces:
            if s not in surface_dict:
                surface_dict[s] = 0
            surface_dict[s] += f 


def get_relation_templates(concept_pairs):
    global surface_dict
    surface_dict = dict() 
    #for subj, obj in concept_pairs:
    #     get_relation(subj, obj)
    with Pool(10) as p:
        p.starmap(get_relation, concept_pairs)
    return surface_dict
