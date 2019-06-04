#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:14:35 2019

@author: Padmanabhan Rajendrakumar
"""

#run using spark-submit --packages org.mongodb.spark:mongo-spark-connector_2.11:2.4.0 clean_load_nyt_articles.py
#PYSPARK_DRIVER_PYTHON=jupyter needs to be changed to PYSPARK_DRIVER_PYTHON=python. Otherwise the program won't run

from pyspark.sql.types import  StringType
from pyspark.ml.feature import StopWordsRemover, Tokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
import nltk
import os
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import udf

#spark = SparkSession.builder.appName('Topic Modelling - Cleaning - NYT').getOrCreate()
spark = SparkSession \
    .builder \
    .appName('Topic Modelling - Cleaning - NYT') \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext
nltk.download('wordnet')
lemma = WordNetLemmatizer()

filename = "nytimes_2000_2019.csv"

news_data = spark.read.csv(filename, header=True)
snippet_text = news_data.select('snippet')
snippet_text = snippet_text.replace(r'\\n\\n|\\n',' ')

def stopwords_read(stopWordsFile):
    f = open(stopWordsFile, "r")
    stopWords = set(f.read().split("\n"))
    return stopWords

def clean_words(text):
    if text == None:
        text = 'news'
    punc_removed = text.translate(str.maketrans('', '', string.punctuation))
    words = punc_removed.split()
    stopwords = stopwords_read("stopwords.txt")
    stopwords = [word.lower() for word in stopwords]    
    lemm_words = [lemma.lemmatize(word,'v') for word in words] #Lemmatizing verbs
    lemm_words = [lemma.lemmatize(word) for word in lemm_words] #Lemmatizing nouns
    lemm_words = [lemma.lemmatize(word,pos='a') for word in lemm_words] #Lemmatizing adjectives
    output_text = [re.sub('[^a-zA-Z0-9]','',word) for word in lemm_words]                                       
    output_text = [word.lower() for word in output_text if len(word)>2 and len(word)<=30 and word.lower() not in stopwords]
    output_text = " ".join(lemma.lemmatize(word,'v') for word in output_text)
    return output_text


udf_clean = udf(clean_words, StringType())
data_cleaned = snippet_text.withColumn("snippet_c", udf_clean(snippet_text['snippet']))
tokenizer = Tokenizer(inputCol="snippet_c", outputCol="tokens")
data_tokenized = tokenizer.transform(data_cleaned)
data_tokenized_2cols = data_tokenized.select("snippet_c", "tokens")
SWR = StopWordsRemover(inputCol="tokens", outputCol="tokens_final")
data_final = SWR.transform(data_tokenized_2cols)
op = data_final.select('tokens_final')

print("Writing to CSV")
op.toPandas().to_csv('nyt_cleaned.csv')

print("Writing to MongoDB")
op.write.format("com.mongodb.spark.sql.DefaultSource").mode("append").save()
