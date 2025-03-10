### BUILDING RECOMMENDER SYSTEMS USING NMF


# Finding Similar Articles
-Engineer at a large online newspaper
-Task: recommend articles similar to article being read by customer
-Similar articles should have similar topics


# Strategy

-Apply NMF to the word-freq array
-NMF Feature values describe the TOPICS
    -... so similar documents have similar NMF feature values
-Compare NMF Feature Values?


# Apply NMF to the word-freq array
-'articles' is a word-freq array

from sklearn.decomposition import NMF
nmf = NMF(n_components=6)
nmf_features = nmf.fit_transform(articles)


# Now we need to define how to compare

-Versions of articles
    -Different versions of the SAME document have same TOPIC proportions
    -... exact feature values may be DIFFERENT!
        -e.g. because one version uses many meaningless words
    -but ALL versions lie on the SAME line through the ORIGIN

-we will use COSINE SIMILARITY comparison
    -use the ANGLE between the lines
    -HIGHER values means MORE similar
    -max val is 1, when angle is 0 degrees


# Calculating the Cosine Similarities

import sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
# if has index 23
current_article = norm_features[23,:]
similarities = norm_features.dot(current_article)
print(similarities)


# DataFrames and Labels
    -label similarities with article titles, using a DataFrame
    -Titles give as a list: titles

import pandas as pd
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
current_article = df.loc['Dog bites man']
similarities = df.dot(current_article)

# Finally

print(similarities.nlargest())
    -to find the articles with the HIGHEST Cosine Similarities
#######################################################################

# PRACTICE

# Which Articles are similar to 'Cristiano Ronaldo'?

In the video, you learned how to use NMF features and the cosine similarity to find similar articles. 
Apply this to your NMF model for popular Wikipedia articles, by finding the articles most similar to the article about the footballer Cristiano Ronaldo. 
The NMF features you obtained earlier are available as nmf_features, while titles is a list of the article titles.


Import normalize from sklearn.preprocessing.
Apply the normalize() function to nmf_features. Store the result as norm_features.
Create a DataFrame df from norm_features, using titles as an index.
Use the .loc[] accessor of df to select the row of 'Cristiano Ronaldo'. Assign the result to article.
Apply the .dot() method of df to article to calculate the cosine similarity of every row with article.
Print the result of the .nlargest() method of similarities to display the most similar articles. This has been done for you, so hit 'Submit Answer' to see the result!

from sklearn.preprocessing import normalize
norm_features = normalize(nmf_features)
df = pd.DataFrame(norm_features, index=titles)
article = df.loc['Cristiano Ronaldo']
similarities = df.dot(article)
print(similarities.nlargest())
###############################################

# Recommend Musical Artists Part 1

In this exercise and the next, you'll use what you've learned about NMF to recommend popular music artists! 
You are given a sparse array artists whose rows correspond to artists and whose columns correspond to users. 
The entries give the number of times each artist was listened to by each user.

In this exercise, build a pipeline and transform the array into normalized NMF features. 
The first step in the pipeline, MaxAbsScaler, transforms the data so that all users have the same influence on the model, 
regardless of how many different artists they've listened to. In the next exercise, you'll use the resulting normalized NMF features for recommendation!


Import:
NMF from sklearn.decomposition.
Normalizer and MaxAbsScaler from sklearn.preprocessing.
make_pipeline from sklearn.pipeline.
Create an instance of MaxAbsScaler called scaler.
Create an NMF instance with 20 components called nmf.
Create an instance of Normalizer called normalizer.
Create a pipeline called pipeline that chains together scaler, nmf, and normalizer.
Apply the .fit_transform() method of pipeline to artists. Assign the result to norm_features.


from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
scaler = MaxAbsScaler()
nmf = NMF(n_components=20)
normalizer = Normalizer()
pipeline = make_pipeline(scaler, nmf, normalizer)
norm_features = pipeline.fit_transform(artists)


# PART 2

Suppose you were a big fan of Bruce Springsteen - which other musical artists might you like? 
Use your NMF features from the previous exercise and the cosine similarity to find similar musical artists. 
A solution to the previous exercise has been run, so norm_features is an array containing the normalized NMF features as rows. 
The names of the musical artists are available as the list artist_names.


Import pandas as pd.
Create a DataFrame df from norm_features, using artist_names as an index.
Use the .loc[] accessor of df to select the row of 'Bruce Springsteen'. Assign the result to artist.
Apply the .dot() method of df to artist to calculate the dot product of every row with artist. Save the result as similarities.
Print the result of the .nlargest() method of similarities to display the artists most similar to 'Bruce Springsteen'.

import pandas as pd
df = pd.DataFrame(norm_features, index=artist_names)
artist = df.loc['Bruce Springsteen']
similarities = df.dot(artist)
print(similarities.nlargest())

#####################################################################