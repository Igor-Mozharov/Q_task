import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

#Convert data in spacy format for training

df = pd.read_csv('db.csv')

train_data = []

for name, sentence in zip(list(df['Mountain']), list(df['Sentence'])):
    start_index = sentence.index(name)
    end_index = start_index + len(name)
    label = 'mount'
    train_data.append((sentence, [(start_index, end_index, label)]))

nlp = spacy.load('en_core_web_sm')

db = DocBin()

for text, annot in tqdm(train_data):
    doc = nlp(text)
    ents = []
    for start, end, label in annot:
        span = doc.char_span(start_idx=start, end_idx=end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk('./train.spacy')

#next step ---> python3 -m spacy init fill-config base_config.cfg config.cfg
#next step ----> python3 -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./train.spacy
