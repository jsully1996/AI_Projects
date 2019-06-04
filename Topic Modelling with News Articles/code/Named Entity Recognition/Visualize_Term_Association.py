import pandas as pd
import spacy
import scattertext as st
from spacy.displacy.render import EntityRenderer
from IPython.core.display import display, HTML
import re


nlp = spacy.load('en_core_web_lg')


df = pd.read_csv('../input/socc_articles.csv')


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', str(raw_html))
    return cleantext


df['clean_article_text'] = df['article_text'].apply(cleanhtml)
df.drop('article_text', axis=1, inplace=True)

df_1 = df.groupby( [ "author"] ).size().reset_index(name='Counts')
df_1 = df_1.sort_values(by=['Counts'], ascending=False)
df_1 = df_1.head(2)


df_2 = df_1.merge(df, on='author', how='inner')

df_2 = df_2.sort_values(by=['Counts'], ascending=False)

build_corpus = st.CorpusFromPandas(df_2, category_col='author', text_col='clean_article_text', nlp=nlp).build()
df_freq = build_corpus.get_term_freq_df()
df_freq['GLOBE EDITORIAL SCORE'] = build_corpus.get_scaled_f_scores('GLOBE EDITORIAL')
df_freq['Jeffrey Simpson Score'] = build_corpus.get_scaled_f_scores('Jeffrey Simpson')

html = st.produce_scattertext_explorer(build_corpus,
          category='GLOBE EDITORIAL',
          category_name='GLOBE EDITORIAL',
          not_category_name='Jeffrey Simpson',
          width_in_pixels=1000,
          metadata=df_2['author'])


open("../output/Top_2_Authors.html", 'wb').write(html.encode('utf-8'))

#visualizing Empath topics and categories instead of terms

build_feats = st.FeatsFromOnlyEmpath()
build_corpus_2 = st.CorpusFromParsedDocuments(df_2,category_col='author', feats_from_spacy_doc=build_feats, parsed_col='clean_article_text').build()
html = st.produce_scattertext_explorer(build_corpus_2,category='GLOBE EDITORIAL',category_name='GLOBE EDITORIAL',not_category_name='Jeffrey Simpson',width_in_pixels=1000,metadata=df_2['author'],use_non_text_features=True,use_full_doc=True,topic_model_term_lists=build_feats.get_top_model_term_lists())

open("../output/Top_2_Authors-Empath.html", 'wb').write(html.encode('utf-8'))