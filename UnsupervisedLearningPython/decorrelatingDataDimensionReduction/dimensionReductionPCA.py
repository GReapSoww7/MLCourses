### DIMENSION REDUCTION WITH PCA


-represents same data, using LESS Features
-important part of machine learning pipelines
    -can be performed with PCA

# Dimension Reduction with PCA
    -PCA Features are in DECREASING order of variance
    -assumes the LOW variance Features are 'noise'
        -and HIGH variance Features are INFORMATIVE

-specify how many Features to keep
    -e.g. PCA(n_components=2)
    -keeps the FIRST 2 PCA Features
        -Intrinsic Dimension is a GOOD choice

# Dimension Reduction of iris dataset

-'samples' = array of iris measurements (4 features)
-'species' = list of iris species numbers

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(samples)

transformed = pca.transform(samples)
print(transformed.shape)
(150, 2)


# Iris Dataset in 2D

import matplotlib.pyplot as plt
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

    -PCA has REDUCED the Dimension to 2
    -RETAINED the 2 PCA Features with HIGHEST Variance
    -Important info PRESERVED: species remain DISTINCT

# Dimesion Reduction with PCA
-PCA discards LOW Variance PCA Features
-assumes the HIGH Variance FEATURES are Informative
-assumption typically holds in practice (e.g. for IRIS)


# Word Frequency Arrays

-ROWS represent documents, COLUMNS represent words
    -entries MEASURE presence of each word in each document
    -... measure using 'tf-idf' (more later)

# Sparse Arrays and csr_matrix
-'Sparse' = most entries are ZERO

-arrays like this are represented by a special type of array:
    scipy.sparse.csr_matrix INSTEAD of NumPy Array

-csr_matrix SAVES spaces by ONLY remembering the NON-ZERO entries

# TruncatedSVD and csr_matrix
-PCA does NOT support csr_matrix

-use scikit-learn TruncatedSVD
    -this performs the SAME transformation as PCA

from sklearn.decomposition import TruncatedSVD
model = TruncatedSVD(n_components=3)
model.fit(documents) # documents is csr_matrix
transformed = model.transform(documents)
########################################################

# PRACTICE

# Dimension Reduction of the Fish Measurements

In a previous exercise, you saw that 2 was a reasonable choice for the "intrinsic dimension" of the fish measurements. 
Now use PCA for dimensionality reduction of the fish measurements, retaining only the 2 most important components.

The fish measurements have already been scaled for you, and are available as scaled_samples.


Import PCA from sklearn.decomposition.
Create a PCA instance called pca with n_components=2.
Use the .fit() method of pca to fit it to the scaled fish measurements scaled_samples.
Use the .transform() method of pca to transform the scaled_samples. Assign the result to pca_features.


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(scaled_samples)
pca_features = pca.transform(scaled_samples)

print(pca_features.shape)
(85, 2)

-We have reduced dimensionality from 6 to 2

###############################################

# A tf-idf word-frequency array

In this exercise, you'll create a tf-idf word frequency array for a toy collection of documents. 
For this, use the TfidfVectorizer from sklearn. 
It transforms a list of documents into a word frequency array, which it outputs as a csr_matrix. 
It has fit() and transform() methods like other sklearn objects.

You are given a list documents of toy documents about pets.

Import TfidfVectorizer from sklearn.feature_extraction.text.
Create a TfidfVectorizer instance called tfidf.
Apply .fit_transform() method of tfidf to documents and assign the result to csr_mat. This is a word-frequency array in csr_matrix format.
Inspect csr_mat by calling its .toarray() method and printing the result. This has been done for you.
The columns of the array correspond to words. Get the list of words by calling the .get_feature_names() method of tfidf, and assign the result to words


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
csr_mat = tfidf.fit_transform(documents)
print(csr_mat.toarray())
words = tfidf.get_feature_names()
print(words)


<script.py> output:
    [[0.51785612 0.         0.         0.68091856 0.51785612 0.        ]
     [0.         0.         0.51785612 0.         0.51785612 0.68091856]
     [0.51785612 0.68091856 0.51785612 0.         0.         0.        ]]
    ['cats', 'chase', 'dogs', 'meow', 'say', 'woof']

#######################################################################

# Clustering Wikipedia Part 1

You saw in the video that TruncatedSVD is able to perform PCA on sparse arrays in csr_matrix format, such as word-frequency arrays. 
Combine your knowledge of TruncatedSVD and k-means to cluster some popular pages from Wikipedia. 
In this exercise, build the pipeline. 
In the next exercise, you'll apply it to the word-frequency array of some Wikipedia articles.

Create a Pipeline object consisting of a TruncatedSVD followed by KMeans. 
(This time, we've precomputed the word-frequency matrix for you, so there's no need for a TfidfVectorizer).

The Wikipedia dataset you will be working with was obtained from here.

Import:
TruncatedSVD from sklearn.decomposition.
KMeans from sklearn.cluster.
make_pipeline from sklearn.pipeline.
Create a TruncatedSVD instance called svd with n_components=50.
Create a KMeans instance called kmeans with n_clusters=6.
Create a pipeline called pipeline consisting of svd and kmeans.

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)

# Part II

It is now time to put your pipeline from the previous exercise to work! 
You are given an array articles of tf-idf word-frequencies of some popular Wikipedia articles, and a list titles of their titles. 
Use your pipeline to cluster the Wikipedia articles.

A solution to the previous exercise has been pre-loaded for you, so a Pipeline pipeline chaining TruncatedSVD with KMeans is available.


Import pandas as pd.
Fit the pipeline to the word-frequency array articles.
Predict the cluster labels.
Align the cluster labels with the list titles of article titles by creating a DataFrame df with labels and titles as columns. This has been done for you.
Use the .sort_values() method of df to sort the DataFrame by the 'label' column, and print the result.
Hit submit and take a moment to investigate your amazing clustering of Wikipedia pages!

import pandas as pd
pipeline.fit(articles)
labels = pipeline.predict(articles)
df = pd.DataFrame({'label': labels, 'article': titles})
print(df.sort_values('label'))



# Part I and Part II Combined Code

from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

svd = TruncatedSVD(n_components=50)
kmeans = KMeans(n_clusters=6)
pipeline = make_pipeline(svd, kmeans)

import pandas as pd
pipeline.fit(articles)
labels = pipeline.predict(articles)
df = pd.DataFrame({'label': labels, 'article': titles})
print(df.sort_values('label'))

<script.py> output:
        label                                        article
    59      0                                    Adam Levine
    57      0                          Red Hot Chili Peppers
    56      0                                       Skrillex
    55      0                                  Black Sabbath
    54      0                                 Arctic Monkeys
    53      0                                   Stevie Nicks
    52      0                                     The Wanted
    51      0                                     Nate Ruess
    50      0                                   Chad Kroeger
    58      0                                         Sepsis
    30      1                  France national football team
    31      1                              Cristiano Ronaldo
    32      1                                   Arsenal F.C.
    33      1                                 Radamel Falcao
    37      1                                       Football
    35      1                Colombia national football team
    36      1              2014 FIFA World Cup qualification
    38      1                                         Neymar
    39      1                                  Franck Ribéry
    34      1                             Zlatan Ibrahimović
    26      2                                     Mila Kunis
    28      2                                  Anne Hathaway
    27      2                                 Dakota Fanning
    25      2                                  Russell Crowe
    29      2                               Jennifer Aniston
    23      2                           Catherine Zeta-Jones
    22      2                              Denzel Washington
    21      2                             Michael Fassbender
    20      2                                 Angelina Jolie
    24      2                                   Jessica Biel
    10      3                                 Global warming
    11      3       Nationally Appropriate Mitigation Action
    13      3                               Connie Hedegaard
    14      3                                 Climate change
    12      3                                   Nigel Lawson
    16      3                                        350.org
    17      3  Greenhouse gas emissions by the United States
    18      3  2010 United Nations Climate Change Conference
    19      3  2007 United Nations Climate Change Conference
    15      3                                 Kyoto Protocol
    8       4                                        Firefox
    1       4                                 Alexa Internet
    2       4                              Internet Explorer
    3       4                                    HTTP cookie
    4       4                                  Google Search
    5       4                                         Tumblr
    6       4                    Hypertext Transfer Protocol
    7       4                                  Social search
    49      4                                       Lymphoma
    42      4                                    Doxycycline
    47      4                                          Fever
    46      4                                     Prednisone
    44      4                                           Gout
    43      4                                       Leukemia
    9       4                                       LinkedIn
    48      4                                     Gabapentin
    0       4                                       HTTP 404
    45      5                                    Hepatitis C
    41      5                                    Hepatitis B
    40      5                                    Tonsillitis
##################################################################################################