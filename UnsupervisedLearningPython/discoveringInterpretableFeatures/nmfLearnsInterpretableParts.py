### NMF LEARNS INTERPRETABLE PARTS


# Example: NMF Learns Interpretable Parts
-word-freq array 'articles' (tf-idf)
-20,000 scientific articles (rows) [components]
-800 words (columns)

print(articles.shape)
(20000, 800)

from sklearn.decomposition import NMF
nmf = NMF(n_components=10)
nmf.fit(articles)

print(nmf.components_.shape)
(10, 800)


# NMF Compnents
-for documents:
    -NMF components represent TOPICS
    -NMF Features combine topics into documents
-for images, NMF components are PARTS of images

# Grayscale Images
-no colors, only shades of gray
    -measure pixel brightness (code it by)
    -represent with a val between 0 and 1 (0 is black)
    -convert to 2D array

-ex.
    -an 8x8 grayscale image written as an array

-then we make it a FLAT ARRAY
    -enumerate entries = flattening
    -row-by-row
    -from left to right, top to bottom

# Encoding a collection of images
-collection of images of the same size
    -can be encoded as 2D array
    -each ROW corresponds to an IMAGE
    -each COLUMN corresponds to a PIXEL
-we can apply NMF


# Visualizing Samples
print(sample)

bitmap = sample.reshape((2,3))
print(bitmap)

import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.show()
##################################################################

# PRACTICE

# NMF learns topics of documents

In the video, you learned when NMF is applied to documents, the components correspond to topics of documents, and the NMF features reconstruct the documents from the topics. 
Verify this for yourself for the NMF model that you built earlier using the Wikipedia articles. 
Previously, you saw that the 3rd NMF feature value was high for the articles about actors Anne Hathaway and Denzel Washington. 
In this exercise, identify the topic of the corresponding NMF component.

The NMF model you built earlier is available as model, while words is a list of the words that label the columns of the word-frequency array.

After you are done, take a moment to recognize the topic that the articles about Anne Hathaway and Denzel Washington have in common!


Import pandas as pd.
Create a DataFrame components_df from model.components_, setting columns=words so that columns are labeled by the words.
Print components_df.shape to check the dimensions of the DataFrame.
Use the .iloc[] accessor on the DataFrame components_df to select row 3. Assign the result to component.
Call the .nlargest() method of component, and print the result. This gives the five words with the highest values for that component.

import pandas as pd
components_df = pd.DataFrame(model.components_, columns=words)
print(components_df.shape)
component = components_df.iloc[3,:]
print(component.nlargest())

################################################################

# Explore the LED Digits Dataset

In the following exercises, you'll use NMF to decompose grayscale images into their commonly occurring patterns. 
Firstly, explore the image dataset and see how it is encoded as an array. 
You are given 100 images as a 2D array samples, where each row represents a single 13x8 image. 
The images in your dataset are pictures of a LED digital display.


Import matplotlib.pyplot as plt.
Select row 0 of samples and assign the result to digit. For example, to select column 2 of an array a, you could use a[:,2]. 
Remember that since samples is a NumPy array, you can't use the .loc[] or iloc[] accessors to select specific rows or columns.
Print digit. This has been done for you. Notice that it is a 1D array of 0s and 1s.
Use the .reshape() method of digit to get a 2D array with shape (13, 8). Assign the result to bitmap.
Print bitmap, and notice that the 1s show the digit 7!
Use the plt.imshow() function to display bitmap as an image.


import matplotlib.pyplot as plt
digit = samples[0,:]
print(digit)
bitmap = digit.reshape(13,8)
print(bitmap)
plt.imshow(bitmap, cmap='gray', interpolation='neartest')
plt.colorbar()
plt.show()

#########################################################

# NMF Learns the Parts of Images

Now use what you've learned about NMF to decompose the digits dataset. 
You are again given the digit images as a 2D array samples. 
This time, you are also provided with a function show_as_image() that displays the image encoded by any 1D array:

def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()
After you are done, take a moment to look through the plots and notice how NMF has expressed the digit as a sum of the components!


Import NMF from sklearn.decomposition.
Create an NMF instance called model with 7 components. (7 is the number of cells in an LED display).
Apply the .fit_transform() method of model to samples. Assign the result to features.
To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.
Assign the row 0 of features to digit_features.
Print digit_features.

from sklearn.decomposition import NMF
model = NMF(n_components=7)
features = model.fit_transform(samples)
for component in model.components_:
    show_as_image(component)

digit_features = features[0,:]
print(digit_features)
############################################################

# PCA DOES NOT LEARN PARTS

Unlike NMF, PCA doesn't learn the parts of things. 
Its components do not correspond to topics (in the case of documents) or to parts of images, when trained on images. 
Verify this for yourself by inspecting the components of a PCA model fit to the dataset of LED digit images from the previous exercise. 
The images are available as a 2D array samples. Also available is a modified version of the show_as_image() function which colors a pixel red if the value is negative.

After submitting the answer, notice that the components of PCA do not represent meaningful parts of images of LED digits!


Import PCA from sklearn.decomposition.
Create a PCA instance called model with 7 components.
Apply the .fit_transform() method of model to samples. Assign the result to features.
To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.


from sklearn.decomposition import PCA
model = PCA(n_components=7)
features = model.fit_transform(samples)
for component in model.components_:
    show_as_image(component)

############################################################################################