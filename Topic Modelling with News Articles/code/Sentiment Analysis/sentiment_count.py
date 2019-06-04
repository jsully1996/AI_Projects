import datetime
import json
import pandas as pd
import numpy as np
import sys

# For plotting
import matplotlib.pyplot as plt

# NLTK's sentiment code
import nltk
import nltk.sentiment.util
import nltk.sentiment.vader
from ast import literal_eval
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
# TextBlob provides its own sentiment analysis
from textblob import TextBlob

def main(data):
    untokenized = pd.read_csv(data)
    polarities =[]
    subjectivities = []
    snips = untokenized['snippet'].tolist()
    for s in snips:
        try:
            polarities.append(TextBlob(s).sentiment[0])
            subjectivities.append(TextBlob(s).sentiment[1])
        except:
            continue
    return polarities
    
def polarity_score(polarities):
    positive_polarity = [p for p in polarities if p>0]
    negative_polarity = [n for n in polarities if n<0]
    neutral_polarity = [r for r in polarities if r==0]
    '''
    hp_polarity = [p for p in positive_polarity if p>2/3]
    hp_size = len(hp_polarity)/len(positive_polarity)
    mop_polarity = [p for p in positive_polarity if p<=2/3 and p>=1/3]
    mop_size = len(mop_polarity)/len(positive_polarity)
    mp_polarity = [p for p in positive_polarity if p<1/3]
    mp_size = len(mp_polarity)/len(positive_polarity)
    hn_polarity = [n for n in negative_polarity if n<-2/3]
    hn_size = len(hp_polarity)/len(negative_polarity)
    mon_polarity = [n for n in negative_polarity if n>=-2/3 and n<=-1/3]
    mon_size = len(mon_polarity)/len(negative_polarity)
    mn_polarity = [n for n in negative_polarity if n>-1/3]
    mn_size = len(mn_polarity)/len(negative_polarity)
    '''
    total_size = len(positive_polarity) + len(negative_polarity) + len(neutral_polarity)
    n_size = len(negative_polarity)/total_size
    p_size = len(positive_polarity)/total_size
    r_size = len(neutral_polarity)/total_size
    return n_size,p_size,r_size

def plot_results(n_size,p_size,r_size):
    main_labels = ['Neutral Reactions', 'Positive Reactions', 'Negative Reactions']
    #sublabels = ['Highly Positive Reactions', 'Moderately Positive Reactions','Mildly Positive Reactions','Highly Negative Reactions', 'Moderately Negative Reactions','Mildly Negative Reactions','']
    sizes = [r_size, p_size, n_size]
    #subsizes = [p_size, hp_size,mop_size,mp_size, n_size]

    #colors
    colors = ['#ffaa00','green','red']
    #explsion
    explode = (0.01,0.01,0.01)
    plt.rcParams['font.size'] = 19.0
    plt.figure(figsize=(30,20))
    plt.pie(sizes, colors = colors, labels=main_labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode,wedgeprops={'alpha':0.55})
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.savefig('reslut_pie.png')
    
if __name__ == '__main__':
   snippet_file = sys.argv[1]
   pol = main(snippet_file)
   n,p,r = polarity_score(pol)
   plot_results(n,p,r)


