from pyspark.sql.types import  StringType
from pyspark.ml.feature import StopWordsRemover, Tokenizer
import string
from nltk.stem.wordnet import WordNetLemmatizer
import re
import sys
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import udf

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

def main(data_file_csv):
    news_data = spark.read.csv(data_file_csv, header=True)
    snippet_text = news_data.select('snippet')
    snippet_text = snippet_text.replace(r'\\n\\n|\\n',' ')
    udf_clean = udf(clean_words, StringType())
    data_cleaned = snippet_text.withColumn("snippet_c", udf_clean(snippet_text['snippet']))
    tokenizer = Tokenizer(inputCol="snippet_c", outputCol="tokens")
    data_tokenized = tokenizer.transform(data_cleaned)
    data_tokenized_2cols = data_tokenized.select("snippet_c", "tokens")
    SWR = StopWordsRemover(inputCol="tokens", outputCol="tokens_final")
    data_final = SWR.transform(data_tokenized_2cols)
    output = data_final.select('tokens_final')
    return output
    
if __name__ == '__main__':
    spark = SparkSession.builder.appName('Extracting - Cleaning - NYT').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    nltk.download('wordnet')
    lemma = WordNetLemmatizer()
    data_file = sys.argv[1]
    tokenfile_name = sys.argv[2]
    op = main(data_file)
    op.toPandas().to_csv(tokenfile_name)

    
    
    
