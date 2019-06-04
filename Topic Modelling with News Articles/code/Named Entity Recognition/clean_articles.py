import pyspark
import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
import os
import re 
import struct
from struct import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StringType, ArrayType
import pandas as pd
from pyspark.ml.feature import StopWordsRemover, Tokenizer, CountVectorizer, HashingTF, IDF
from pyspark.sql.types import ArrayType, StringType


spark = SparkSession.builder.appName("spark-clean").getOrCreate()
         
         
data = spark.read.csv("../input/socc_articles.csv", header=True) #gives a dataframe
sc = spark.sparkContext

def cleanhtml(raw_html):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', str(raw_html))
	return cleantext

comment_text = data.select('article_text')
new_func = udf(lambda x: cleanhtml(x))
cleaned = comment_text.withColumn("clean_article_text", new_func(comment_text.article_text))
cleaned = cleaned.select("clean_article_text")
cleaned = cleaned.na.drop()

# cleaned = cleaned.limit(1)
cleaned = cleaned.toPandas()
cleaned.to_csv("../output/cleaned_articles.csv")
