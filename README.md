# rec_sys
My recommender system uses the movielens rating data from http://grouplens.org/datasets/movielens/
It then performs the following steps:

1.	Populates movie and user dictionaries and calculate rating predictions based on average
2.	Creates two new dictionaries:
    I.	A movie prediction dictionary with predictions made by using the Pearson’s correlation between users to weight the prediction values
    II.	A squared error dictionary with squared errors between the actual rating for that user and the prediction.
3.	Creates a similarity matrix that compares each user against all others based on the ratings they have given to movies they have co-rated. The matrix contains a value for each user pair with a similarity using the Pearson Pearson’s Correlation which measures the extent to which two variables are linear related; -1 perfect negative correlation, 0 no correlation, 1  perfect correlation.  See figure 1.
4.	Creates a matrix with a row for each user per movie with the actual rating, predicted and the mean-squared error for each predicted value (where a prediction can be made). Predictions cannot be made if the current user is the only rater of the movie, a null value is assigned. 
5.	Converts the matrix to a CSV file 
6.	Calculates the coverage metric which is the percentage of number of valid predictions divided by the total number of reviews (100000).
7.	Calculate an overall root-mean squared error which is the square root of the mean of all the squared difference scores
8.	Calculates the average run-time to make all predictions based on 10 iterations
