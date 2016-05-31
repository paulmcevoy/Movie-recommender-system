# -*- coding: utf-8 -*-

'''
#-------------------WEEK1--------------------------------------------------------------

The main.py program runs the following steps:
1.	Read in the 100k.dat file line by line 
2.	Each line is packed into a dictionary with tuple headings indicating movie_id, user_id etc.
3.	Two dictionaries and one list are created: 
      I.	Python dictionary movie_dict referenced by a movie_id as key. 
      Within each movie entry is a user_id/rating dictionary referenced by user_id (see Figure 1 as an example)
      II.	Python dictionary user_dict referenced by a user_id as key. 
      Within each user entry is a movie_id/rating dictionary referenced by movie_id
      III.	A python list of all ratings, ratings_list
4.	The statistics for each user, movie and rating is output to CSV files in the local directory
'''

#import the functions from rec_funcs
from rec_funcs import get_stats, show_stats_wk1, dict_ratings
from rec_funcs import csv_output, calc_sim, create_sim_matrix, rmse_timer
#-------------------WEEK1--------------------------------------------------------------

#Populate each of the dictionaries
dict_ratings()
#Populate each of the stats dictionaries
#user_stats, movie_stats = get_stats()
#print the stats
#show_stats_wk1(user_stats, movie_stats)
#output the stats to CSV files
#csv_output(user_stats, movie_stats)
'''
#-------------------WEEK2--------------------------------------------------------------

1.	All steps from Week1 to populate movie and user dictionaries
2.	Creates an empty matrix (100000 rows x 5 columns)
3.	Populates the matrix with:
    a.	The user_id
    b.	The movie_id
    c.	The actual rating for that user_id for that movie_id
    d.	The average rating review for that movie_id across all users without the current user
    e.	The Root-Mean-Squared-Error
4.	The RMSE value is only calculated when the size of the rating list is > 0. ie, if the current user is the only reviewer for that movie then a prediction can’t be made.
5.	The matrix is output to a CSV (rmse.csv). See Figure 1 for snippet example
6.	A coverage metric indicating the percentage of predictions that were made from the total number of ratings
7.	A mean RMSE is calculated as the average RMSE across all available ratings
8.	Performance is calculated as the average time 10 loops of the RMSE matrix generation
'''
from rec_funcs import create_pred_dict, create_rmse_dict, show_rmse_info_neb, show_rmse_info_ave, create_pred_dict_neb, create_rmse_dict, create_stats_csv

#show_stats_wk2(rmse_matrix)
#create_pred_dict('fast')
#rmse_timer()

'''
#-------------------WEEK3--------------------------------------------------------------
1.	All steps from Week1 and Week2 to populate movie and user dictionaries and calculate rating predictions based on average
2.	Creates two new dictionaries:
    I.	A movie prediction dictionary with predictions made by using a specific number of similar neighbours of other users that rated that movie
    II.	A squared error dictionary with squared errors between the actual rating for that user and the prediction.
3.	Creates a similarity matrix that compares each user against all others based on the ratings they have given to movies they have co-rated. The matrix contains a value for each user pair with a similarity between 0 and 1 (1 indicating perfect similarity) See figure 1.
4.	Creates a matrix with a row for each user per movie with the actual rating, predicted and the mean-squared error for each predicted value (where a prediction can be made). Predictions cannot be made if the current user is the only rater of the movie or there are not enough neighbours that have rated that movie, a null value is placed in the matrix. (See figure 2)
5.	Converts the matrix to a CSV file for output for each value ‘n’ of number of neighbours
6.	Calculates the coverage metric which is the percentage of number of valid predictions divided by the total number of reviews (100000).
7.	Calculate an overall root-mean squared error which is the square root of the mean of all the RMSE scores
8.	Calculates the average run-time to make all predictions based on 10 iterations

'''
#create the similarity matrix
#create_sim_matrix()
#from rec_funcs import user_dict, movie_dict, pred_dict, sim_matrix

#for neb_size in (2, 5,10,15,20,50, 80, 100):
#    create_pred_dict_neb(neb_size)
#    create_rmse_dict_neb()
#    show_rmse_info_neb(neb_size)
#create_stats_csv()    

#rmse_timer('neighbour')
'''
#-------------------WEEK4--------------------------------------------------------------
1.	All steps from Week1 and Week2  and week 3 to populate movie and user dictionaries and calculate rating predictions based on average
2.	Creates two new dictionaries:
    I.	A movie prediction dictionary with predictions made by using the cosine distance between users to weight the prediction values
    II.	A squared error dictionary with squared errors between the actual rating for that user and the prediction.
3.	Creates a similarity matrix that compares each user against all others based on the ratings they have given to movies they have co-rated. The matrix contains a value for each user pair with a similarity using the cosine distance between 0 and 1 (1 indicating perfect similarity) See figure 1.
4.	Creates a matrix with a row for each user per movie with the actual rating, predicted and the mean-squared error for each predicted value (where a prediction can be made). Predictions cannot be made if the current user is the only rater of the movie, a null value is placed in the matrix. (See figure 2)
5.	Converts the matrix to a CSV file for output for 
6.	Calculates the coverage metric which is the percentage of number of valid predictions divided by the total number of reviews (100000).
7.	Calculate an overall root-mean squared error which is the square root of the mean of all the RMSE scores
8.	Calculates the average run-time to make all predictions based on 10 iterations
'''
'''
from rec_funcs import create_pred_dict_cos, show_rmse_info_cos
main_this_time = time.time()
create_sim_matrix('cos')
create_pred_dict_cos()
create_rmse_dict('cos')
show_rmse_info_cos()

'''
#for viewing locally


#from rec_funcs import user_dict, movie_dict, pred_dict, rmse_dict, rmse_list_ave, rmse_list_neb, sim_matrix, pred_dict_neb, stats_list
#from rec_funcs import pred_dict_cos, rmse_dict_cos, rmse_df

'''
#-------------------WEEK5--------------------------------------------------------------
1.	All steps from Week1 and Week2, Week 3 and Week4 to populate movie and user dictionaries and calculate rating predictions based on average
2.	Creates two new dictionaries:
    I.	A movie prediction dictionary with predictions made by using the Pearson’s correlation between users to weight the prediction values
    II.	A squared error dictionary with squared errors between the actual rating for that user and the prediction.
3.	Creates a similarity matrix that compares each user against all others based on the ratings they have given to movies they have co-rated. The matrix contains a value for each user pair with a similarity using the Pearson Pearson’s Correlation which measures the extent to which two variables are linear related; -1 perfect negative correlation, 0 no correlation, 1  perfect correlation.  See figure 1.
4.	Creates a matrix with a row for each user per movie with the actual rating, predicted and the mean-squared error for each predicted value (where a prediction can be made). Predictions cannot be made if the current user is the only rater of the movie, a null value is assigned. 
5.	Converts the matrix to a CSV file 
6.	Calculates the coverage metric which is the percentage of number of valid predictions divided by the total number of reviews (100000).
7.	Calculate an overall root-mean squared error which is the square root of the mean of all the squared difference scores
8.	Calculates the average run-time to make all predictions based on 10 iterations

'''
from rec_funcs import create_ave_dict, user_dict, movie_dict, user_ave_dict, get_pred_res, create_pred_dict_res, create_rmse_dict, show_rmse_info_res

create_ave_dict()
create_sim_matrix('res')
create_pred_dict_res()
create_rmse_dict('res')
show_rmse_info_res()

from rec_funcs import sim_matrix
from rec_funcs import pred_dict_res, rmse_dict_res
