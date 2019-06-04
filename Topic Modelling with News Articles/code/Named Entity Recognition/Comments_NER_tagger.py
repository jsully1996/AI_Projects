import pandas as pd
import spacy
from spacy.displacy.render import EntityRenderer


# pd.set_option('display.max_rows', 10) 
# pd.options.mode.chained_assignment = None 

nlp = spacy.load('en_core_web_lg')


df = pd.read_csv('../input/socc_comments.csv')

lower = lambda x: x.lower() # make everything lowercase

df = pd.DataFrame(df['comment_text'].apply(lower))
df.columns = ['text']
#display(df)

def extract_named_ents(text):
    return [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]

def add_named_ents(df):
    df['named_ents'] = df['text'].apply(extract_named_ents)  
add_named_ents(df)
df.to_csv('../output/comments_NER.csv')
final = df['named_ents']

final_1 = final.values.flatten()
flattened_list = [y for x in final_1 for y in x]
df_1 = pd.DataFrame({'col':flattened_list})

df_1['Entity'], df_1['Tags'] = df_1['col'].str[0], df_1['col'].str[3]

df_2 = df_1.groupby( [ "Entity", "Tags"] ).size().reset_index(name='Counts')
final_df = df_2.sort_values(by=['Counts'], ascending=False)
top_in_each_df = final_df.loc[final_df.groupby('Tags').Counts.agg('idxmax')]
top_in_each_df.to_csv('../output/comments_top_in_cat.csv')