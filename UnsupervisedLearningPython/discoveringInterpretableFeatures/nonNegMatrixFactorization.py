### NON-NEGATIVE MATRIX FACTORIZATION (NMF)

-is a Dimension reduction technique

-NMF models are interpretable (UNLIKE PCA)
    -ALL sample Features must be NON-NEGATIVE

# Interpretable Parts
-NMF expresses documents as combinations of topics (or "themes")
-expresses images as combinations of patterns

# using scikit-learn NMF

-follows fit() & transform() pattern
-must SPECIFY number of components e.g NMF(n_components=2)
-works with NumPy arrays and with csr_matrix

# Example Word-Frequency Array

-word freq array, 4 words, many documents
-measure presence of words in each document using 'tf-idf'
    -'tf' = frequency of word in document
    -'idf' = reduces the influence of frequent words


# Example usage of NMF

-'samples' is the word-freq array

from sklearn.decomposition import NMF
model = NMF(n_components=2)
model.fit(samples)
nmf_features = model.transform(samples)


# NMF Components
-Dimension of components = dimension of samples


# NMF Features
-can be used to reconstruct the samples
    -combine features with components

# Reconstruction of a sample

print(samples[i,:])
[ 0.12 0.18 0.32 0.14 ]

print(nmf_features[i,:])
[ 0.15 0.12 ]

# Sample Reconstruction
-MULTIPLE components by Feature VALUES, and ADD up
-can be expressed as a PRODUCT of matrices
-this is the 'Matrix Factorization' in 'NMF'


# NMF fits into non-negative data ONLY
-word frequencies in each document
-images encoded as arrays
-audio spectrograms
-purchase histories on e-commerce sites


#############################


# PRACTICE

# NMF applied to Wikipedia articles Part 1

In the video, you saw NMF applied to transform a toy word-frequency array. 
Now it's your turn to apply NMF, this time using the tf-idf word-frequency array of Wikipedia articles, given as a csr matrix articles. 
Here, fit the model and transform the articles. In the next exercise, you'll explore the result.

Import NMF from sklearn.decomposition.
Create an NMF instance called model with 6 components.
Fit the model to the word count data articles.
Use the .transform() method of model to transform articles, and assign the result to nmf_features.
Print nmf_features to get a first idea what it looks like (.round(2) rounds the entries to 2 decimal places.)

from sklearn.decomposition import NMF
model = NMF(n_components=6)
model.fit(articles)
nmf_features = model.transform(articles)
print(nmf_features.round(2))

# Part 2

Now you will explore the NMF features you created in the previous exercise. 
A solution to the previous exercise has been pre-loaded, so the array nmf_features is available. 
Also available is a list titles giving the title of each Wikipedia article.

When investigating the features, notice that for both actors, the NMF feature 3 has by far the highest value. 
This means that both articles are reconstructed using mainly the 3rd NMF component. 
In the next video, you'll see why: NMF components represent topics (for instance, acting!).


Import pandas as pd.
Create a DataFrame df from nmf_features using pd.DataFrame(). Set the index to titles using index=titles.
Use the .loc[] accessor of df to select the row with title 'Anne Hathaway', and print the result. These are the NMF features for the article about the actress Anne Hathaway.
Repeat the last step for 'Denzel Washington' (another actor).

import pandas as pd
df = pd.DataFrame(nmf_features, index=titles)
print(df.loc['Anne Hathaway'])
print(df.loc['Denzel Washington'])
############################################################