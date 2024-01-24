import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if pd.isnull(text) or isinstance(text, float):
        text = ''
    else:
        text = str(text)
    
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

inverted_index = {}

for i in range(1, 90308):
    file_name = f'{str(i)}.txt'
    
    with open(f'cleaned_data_set/{file_name}', 'r', encoding='utf-8') as file:
        text = file.read()

    doc_id = f'{i}'
    
    preprocessed_text = preprocess(text)
    
    tokens = preprocessed_text.split()
    
    for token in set(tokens):
        if token not in inverted_index:
            inverted_index[token] = [doc_id]
        else:
            inverted_index[token].append(doc_id)
    
    if i % 1000 == 0:
        print(f"{i} documents processed.")

inverted_index = {token: sorted(list(set(doc_ids))) for token, doc_ids in inverted_index.items()}

sorted_inverted_index = dict(sorted(inverted_index.items()))

with open('inverted_index.json', 'w') as file:
    json.dump(sorted_inverted_index, file)

print("Inverted index created and saved to 'inverted_index.json'.")
