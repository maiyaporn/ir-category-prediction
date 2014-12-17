# Categories Prediction
Information Retrieval Final Project (IUB 2014)

## Problem
Task 1-How can we predict a restaurant's categories from a given review text?<br>
Task 2-How to predict user review's rating based on review's Text?

## Dataset
Yelp Dataset Challenge

## Method Task 1
An adaptation from the language model in Information Retrieval where each document is represented by topic distributions.
We use Latent Dirichlet Allocation (LDA), a topic modelling, to find topic distributions from review texts for each category.
The assumption is that a category document is a mixture of topics and the distribution over topics for a document can represent the categories of business instead of a bag of word model.

Cosine Similarity and Hellinger Distance are used in calculating similarity between documents.

## Method Task 2
Used the machine learning approach for predicting the users rating based on the review text. Formulated the features for all the reviews based on one partiular user. Features comprised of the sentiments in the reviews which were analysed and fromulated using Stanford NLP sentimental analysis tool. The training data was trained using J48 algorithm and then the testing data was used to evaluate the approach.

Evaluations were carried out using metrics like RMSE, precision, recall and accuracy.



