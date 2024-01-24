# Information Retrieval System

## Overview and components and functionalities

This project implements an Information Retrieval System using the TF-IDF model and cosine similarity. It consists of two main components: **Indexing** and **Document Retrieval**.

## Indexing
The indexing.py script processes a collection of documents, creating an inverted index for efficient retrieval. Each document is preprocessed using the NLTK library, and the resulting inverted index is saved to inverted_index.json. The indexing is executed in two main steps:

Text Preprocessing: Documents are tokenized, converted to lowercase, and stop words are removed.
Inverted Indexing: Unique tokens are extracted, and an inverted index is created, mapping each token to the corresponding document IDs.

## Document Retrieval

The `retrieval.py` script enables users to input queries for document retrieval. The system uses the inverted index to identify relevant documents based on cosine similarity. The retrieval process involves the following steps:

1. **User Query:** The user enters a query.
2. **Query Preprocessing:** The query undergoes preprocessing similar to the document indexing phase.
3. **Cosine Similarity Calculation:** TF-IDF vectors are calculated for the query and documents, and cosine similarity is used for ranking.
4. **Feedback Mechanism:** Users provide feedback on document relevance, influencing future retrievals.
5. **Evaluation metrics:** 11 point interpolated curve is plotted based on the feedback given 

## Prerequisites

* After unzipping into a folder of your choice enter into the ir_final directory present in the unzipped folder.
* Once here open this directory in vs code

## Running the System

## Steps

step-1 : First we need to extract the folder that is in `src` folder

step-2 : Next we need to download the dataset from the link given below

step-3 : We need to place the `cleaned_data_set` folder in the same
directory of `indexing.py`

step-4 : Now run this command

``` bash
python indexing.py
```

step-4 : Then it will create a `inverted_index.json` file containing
inverted index for the dataset but creating an inverted indexing file
will take long time So i am also uploading the `inverted_index.json`
file in the drive and providing the same link below

step-5 : Now place the `inverted_index.json` file in the same folder of
`retreival.py`

step-6 : Now run the command

``` bash
python retreival.py
```

step-7 : Then it will run the program and then you will be asked for
query (Enter cricketer name and you will get details)

step-8 : Then after displaying the top-5 documents , It will be asking
feedback Then provide feedback

step-9 : After feedback it will display the documents based on feedback
and then it will also display
`11- point interpolated curve for evaluation metrics`

## libraries

We need to download `scikit-learn` , `nltk` , `matplotlib` ,`numpy` for
this we can run this command in terminal

``` bash
pip install scikit-learn nltk matplotlib numpy
```

And also we need to download NLTK data To get into python environment we
need to run this command in terminal and press enter

``` bash
python 
```

for that we can run this command in python environment

``` bash
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## links

Here I am providing the links for both `dataset` and
`inverted_index.json` files

`cleaned_data_set.zip` :
https://drive.google.com/file/d/1oaXkKr3fxB9Durq1xkN0kao5Xv5kASIF/view?usp=sharing

`inverted_index.json` :
https://drive.google.com/file/d/1ICh67vTdbSDVevYtfFLvGGg9X1Sb89A-/view?usp=sharing
