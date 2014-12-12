# Categories Prediction
Information Retrieval Final Project (IUB 2014)

## Problem
How can we predict a restaurant's categories from a given review text?

## Dataset
Yelp Dataset Challenge

## Method
An adaptation from the language model in Information Retrieval where each document is represented by topic distributions.
We use Latent Dirichlet Allocation (LDA), a topic modelling, to find topic distributions from review texts for each category.
The assumption is that a category document is a mixture of topics and the distribution over topics for a document can represent the categories of business instead of a bag of word model.

Cosine Similarity and Hellinger Distance are used in calculating similarity between documents.

More details in presentation.pdf



