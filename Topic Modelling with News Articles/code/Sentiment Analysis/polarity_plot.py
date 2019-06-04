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
    return polarities,subjectivities
    
def plot_polarity(polarities,subjectivities):
    plt.figure(figsize=(30,20))
    plt.scatter(polarities, subjectivities, c=polarities, s=80, cmap='seismic',alpha=0.3)
    plt.xlabel('Article polarity',fontsize=26)
    plt.ylabel('Article subjectivity',fontsize=26)
    plt.grid()
    plt.xlim(-1.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.savefig('polarity_scatterplot.png')

if __name__ == '__main__':
   snippet_file = sys.argv[1]
   p,s = main(snippet_file)
   plot_polarity(p,s)
