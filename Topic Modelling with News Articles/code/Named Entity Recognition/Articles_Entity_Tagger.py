
import pandas as pd
import spacy
from spacy.displacy.render import EntityRenderer
from IPython.core.display import display, HTML

nlp = spacy.load('en_core_web_lg')


df = pd.read_csv('../output/cleaned_articles.csv')


pd.set_option('display.max_rows', 10) 
pd.options.mode.chained_assignment = None 

lower = lambda x: x.lower()

df = pd.DataFrame(df['clean_article_text'].apply(lower))
df.columns = ['text']
display(df)


def extract_named_ents(text):
    return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]

def add_named_ents(df):
    df['named_ents'] = df['text'].apply(extract_named_ents) 

add_named_ents(df)

final = df['named_ents']

final_1 = final.values.flatten()

flattened_list = [y for x in final_1 for y in x]

df_1 = pd.DataFrame({'col':flattened_list})

df_1['Entity'], df_1['Tags'] = df_1['col'].str[0], df_1['col'].str[3]

df_2 = df_1.groupby( [ "Entity", "Tags"] ).size().reset_index(name='Counts')

top_in_each_df = df_2.loc[df_2.groupby('Tags').Counts.agg('idxmax')]

top_in_each_df.to_csv('../output/articles_top_in_cat.csv')

Tag_grp = df_1.groupby( ["Tags"] ).size().reset_index(name='Counts')

Tag_grp.to_csv('../output/tag_counts.csv')


