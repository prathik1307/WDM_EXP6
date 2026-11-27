### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 27.11.26
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:

```
    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download required NLTK packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Student Details
print("Name :Prathiksha R")
print("Register No : 212224040244")

# Sample documents
documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower())

    tokens = [
        token for token in tokens
        if token not in stopwords.words("english")
        and token not in string.punctuation
    ]

    return " ".join(tokens)

# Preprocess documents
preprocessed_docs = {
    doc_id: preprocess_text(doc)
    for doc_id, doc in documents.items()
}

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(
    preprocessed_docs.values()
)

# Search function
def search(query):
    processed_query = preprocess_text(query)

    query_vector = tfidf_vectorizer.transform(
        [processed_query]
    )

    similarity_scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    results = [
        (doc_id, documents[doc_id], score)
        for doc_id, score in zip(
            documents.keys(), similarity_scores
        )
    ]

    return sorted(
        results,
        key=lambda x: x[2],
        reverse=True
    )

# Get user query
query = input("Enter your query: ")

# Perform search
search_results = search(query)

print("Query:", query.upper())

# Display ranked results
for i, result in enumerate(search_results, start=1):

    print(f"\nRank: {i}")
    print("Document ID:", result[0])
    print("Document:", result[1])
    print("Similarity Score:", result[2])
    print("----------------------")

# Highest cosine similarity score
highest_rank_score = search_results[0][2]

print("The highest rank cosine score is:",
      highest_rank_score)
```

### Output:
<img width="830" height="765" alt="image" src="https://github.com/user-attachments/assets/e137c143-e4d1-40be-9682-ecb35c5b3aa1" />

### Result:
Thus, the Information Retrieval System using the Vector Space Model was successfully implemented in Python. The documents were preprocessed, a TF-IDF matrix was constructed, and cosine similarity was calculated between the search query and the documents. The documents were ranked based on their similarity scores, and the most relevant documents were retrieved successfully.
