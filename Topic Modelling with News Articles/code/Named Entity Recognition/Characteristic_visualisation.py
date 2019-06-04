import pandas as pd
import spacy
import scattertext as st
import re
from spacy.displacy.render import EntityRenderer


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


build_corpus = (st.CorpusFromPandas(df_2,
                              category_col='author',
                              text_col='clean_article_text',
                              nlp=st.whitespace_nlp_with_sentences)
          .build()
          .get_unigram_corpus()
          .compact(st.ClassPercentageCompactor(term_count=2,
                                               term_ranker=st.OncePerDocFrequencyRanker)))

html = st.produce_characteristic_explorer(
    build_corpus,
    category='GLOBE EDITORIAL',
    category_name='GLOBE EDITORIAL',
    not_category_name='Jeffrey Simpson',
    metadata=build_corpus.get_df()['author'])

open('../output/characteristic_chart.html', 'wb').write(html.encode('utf-8'))