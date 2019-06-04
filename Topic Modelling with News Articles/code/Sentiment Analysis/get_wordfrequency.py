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

def get_frequency(df):
    data = pd.read_csv(df)
    token_list = list(map(literal_eval,data['tokens_final']))
    token_superlist = [item for sublist in token_list for item in sublist]
    fdist = FreqDist(token_superlist)
    return fdist

def main(fdist):
    fdist.plot(30,cumulative=False)
    plt.savefig('word_frequency.png')
    

if __name__ == '__main__':
    data_file = sys.argv[1]
    f = get_frequency(data_file)
    main(f)


    
