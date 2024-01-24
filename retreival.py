import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import precision_recall_curve
import string
import matplotlib.pyplot as plt
import numpy as np

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]
    processed_text = ' '.join(tokens)
    return processed_text

with open('inverted_index.json', 'r') as file:
    inverted_index = json.load(file)

query = input("Enter the query: ")
query = preprocess_text(query)
query_terms = query.split()
relevant_docs = set()

for term in query_terms:
    if term in inverted_index:
        relevant_docs.update(inverted_index[term])

def read_document(doc_id):
    file_path = f"cleaned_data_set/{doc_id}.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

doc_texts = {doc_id: read_document(doc_id) for doc_id in relevant_docs}
preprocessed_doc_texts = {doc_id: preprocess_text(doc_text) for doc_id, doc_text in doc_texts.items()}
all_texts = list(preprocessed_doc_texts.values()) + [query]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_texts)
doc_tfidf = tfidf_matrix[:-1]
query_tfidf = tfidf_matrix[-1]

cosine_similarities = cosine_similarity(query_tfidf, doc_tfidf).flatten()
doc_similarity = {doc_id: score for doc_id, score in zip(relevant_docs, cosine_similarities)}
sorted_docs = sorted(doc_similarity.items(), key=lambda x: x[1], reverse=True)

print("Top 5 Documents based on Cosine Similarity:")
for rank, (doc_id, score) in enumerate(sorted_docs[:5]):
    print(f"Rank {rank + 1} - Document ID: {doc_id}, Score: {score}")
    print(read_document(doc_id))
    print("=" * 50)

print("\nProvide feedback for the documents:")
feedback = []
rank = 0
while rank < 5 and rank < len(sorted_docs):
    doc_id, _ = sorted_docs[rank]
    user_feedback = input(f"Is Document {doc_id} relevant? (yes/no): ").lower()
    if user_feedback == 'yes':
        feedback.append(doc_id)
    rank += 1

print("\nRelevant Documents based on Feedback:")
for doc_id in feedback:
    print(f"Document ID: {doc_id}")
    print(read_document(doc_id))
    print("=" * 50)

print("\nRemaining Documents:")
for doc_id, score in sorted_docs[5:]:
    print(f"Document ID: {doc_id}, Score: {score}")
    print(read_document(doc_id))
    print("=" * 50)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(np.array([doc_id in feedback for doc_id, _ in sorted_docs]), 
                                               [score for _, score in sorted_docs])

# Calculate 11-point interpolated curve
interp_precision = []
interp_recall = []
recall_levels = np.linspace(0, 1, 11)
for level in recall_levels:
    max_precision = max(prec for rec, prec in zip(recall, precision) if rec >= level)
    interp_precision.append(max_precision)
    interp_recall.append(level)

# Plot the precision-recall curve
plt.plot(recall, precision, marker='o', label='Precision-Recall Curve')
plt.plot(interp_recall, interp_precision, marker='o', linestyle='dashed', label='11-Point Interpolated Curve')
plt.title('Precision-Recall Curve with 11-Point Interpolated Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()
plt.show()
