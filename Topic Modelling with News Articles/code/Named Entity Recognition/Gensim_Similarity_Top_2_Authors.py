import spacy
import pandas as pd
import scattertext as st
import re
from spacy.displacy.render import EntityRenderer
from scattertext import SampleCorpora, word_similarity_explorer_gensim, Word2VecFromParsedCorpus
from scattertext.CorpusFromParsedDocuments import CorpusFromParsedDocuments
from gensim.models import word2vec

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

df_2['parsed'] = df_2.clean_article_text.apply(nlp)
build_corpus = CorpusFromParsedDocuments(df_2, category_col='author', parsed_col='parsed').build()

build_model = word2vec.Word2Vec(size=300,alpha=0.025,window=5,min_count=5,max_vocab_size=None,
                          sample=0,
                          seed=1,
                          workers=1,
                          min_alpha=0.0001,
                          sg=1,
                          hs=1,
                          negative=0,
                          cbow_mean=0,
                          iter=1,
                          null_word=0,
                          trim_rule=None,
                          sorted_vocab=1)

html = word_similarity_explorer_gensim(build_corpus,
                                       category='GLOBE EDITORIAL',
                                       category_name='GLOBE EDITORIAL',
                                       not_category_name='Jeffrey Simpson',
                                       target_term='obama',
                                       minimum_term_frequency=100,
                                       pmi_threshold_coefficient=4,
                                       width_in_pixels=1000,
                                       metadata=df_2['author'],
                                       word2vec=Word2VecFromParsedCorpus(build_corpus, build_model).train(),
                                       max_p_val=0.05,
                                       save_svg_button=True)

open('../output/gensim_similarity_top_2_authors.html', 'wb').write(html.encode('utf-8'))
