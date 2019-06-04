#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:50:26 2019

@author: Padmanabhan Rajendrakumar
"""

import os
import pandas as pd

#Before running this sign up for a https://developer.nytimes.com/ account and add the key as environment variable

my_key = os.environ['my_key']

URL_list = []
start_year = 2000
end_year = 2020

for year in range(start_year,end_year):
    for month in range(1,13):
        url = "https://api.nytimes.com/svc/archive/v1/"+str(year)+"/"+str(month)+".json?api-key="+my_key
        URL_list.append(url)
        if year == 2019 and month == 3:
            break
        

for index, URL in enumerate(URL_list):
    if index == 0:
        print("Fetching from",URL)
        df = pd.read_json(URL)
        news_df = pd.DataFrame(df['response'][0])
        news_df = news_df[['_id','snippet']]
        print("Total No. of News Articles:",len(news_df))
    else:
        print("Fetching from",URL)
        df = pd.read_json(URL)
        news_df_temp = pd.DataFrame(df['response'][0])
        news_df_temp = news_df_temp[['_id','snippet']]
        news_df = news_df.append(news_df_temp)
        print("Total No. of News Articles:",len(news_df))

filename = "nytimes_2000_2019.csv" 

news_df.to_csv(filename, index=False)
