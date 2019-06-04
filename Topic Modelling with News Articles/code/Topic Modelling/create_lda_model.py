#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:05:13 2019

@author: Padmanabhan Rajendrakumar
"""


from pyspark.sql import SparkSession
import gensim
import gensim.corpora as corpora


# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

import ast

spark = SparkSession \
    .builder \
    .appName('Topic Modelling - Cleaning - NYT') \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1/nyt_db.nyt_coll") \
    .getOrCreate()

spark.sparkContext.setLogLevel('WARN')
sc = spark.sparkContext

print("Reading from MongoDB...")
nyt_df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

print("Number of articles:",nyt_df.count())

data = nyt_df.toPandas()
doc_clean = data['tokens_final']
doc_clean = map(str, doc_clean) #For mongo
doc_clean1 = [ast.literal_eval(doc) for doc in doc_clean]

dictionary = corpora.Dictionary(doc_clean1)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean1]

# Creating the object for LDA model using gensim library
Lda = gensim.models.LdaMulticore

# Running and Trainign LDA model on the document term matrix.
print("Creating the LDA model...")
ldamodel = Lda(doc_term_matrix, num_topics=25, id2word = dictionary, passes=50, workers=4)
print("Saving the LDA model...")
ldamodel.save("ldamodel_nyt")
print("LDA model saved...")
