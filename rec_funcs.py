#!/usr/bin/python
from collections import namedtuple, defaultdict, Counter
import numpy as np
import math
import pandas as pd
import time
import csv
from matplotlib import pyplot

movie_dict = defaultdict(dict)
user_dict = defaultdict(dict)
rate_list = []
rating_stats = []
rmse_df = pd.DataFrame()
sim_matrix_size = (943,943)
sim_matrix = np.zeros(sim_matrix_size)
pred_dict = defaultdict(dict)
pred_dict_neb = defaultdict(dict)
pred_dict_cos = defaultdict(dict)
user_ave_dict = defaultdict(dict)
pred_dict_res = defaultdict(dict)
rmse_dict_res = defaultdict(dict)


rmse_dict = defaultdict(dict)
rmse_dict_neb = defaultdict(dict)
rmse_dict_cos = defaultdict(dict)

rmse_list_ave = []
rmse_list_neb = []
rmse_list_cos = []
rmse_list_res = []

movie_list1 = []
movie_list2 = []
csv_file = '100k.dat'
#csv_file = '15.dat'
stats_list = []
dat_len = len(list(csv.reader(open(csv_file, 'rt'), delimiter=',')))
print("File length is {}".format(dat_len))
start_time = time.time()


def movie_parse():
    # Create a matrix from the CSV data
    MovieLensRating = namedtuple('MovieLens', 'user_id, item_id, rating, timestamp')
    for item in map(MovieLensRating._make, csv.reader(open(csv_file, 'rt'), delimiter=',')):
        yield item
        
def dict_ratings():
    #For each user give the movies with the rating they gave
    print("Getting ratings and creating dicts....")
    for line in movie_parse():
        user_dict[int(line.user_id)][int(line.item_id)] = int(line.rating)
        movie_dict[int(line.item_id)][int(line.user_id)] = int(line.rating)
        rate_list.append(line.rating) 
        
        #print(movie_dict)
        
def get_maths(dict_arg):
    #generic function that takes a dict, gets the mean, median, min, max and std_dev
    #and returns a dict using the movie or user as the key
    stat_dict = defaultdict(dict)
    for line in dict_arg:
        rating_list = []
        #line is a dict for each user or movie
        for val in dict_arg[line]:
            rating_list.append(dict_arg[line][val])
        #num_reviews is either the num of reviews per USER or the 
        #number of reviews per MOVIE
        stat_dict[line]['num_reviews'] = len(rating_list)
        stat_dict[line]['mean'] = np.mean(rating_list)
        stat_dict[line]['median'] = np.median(rating_list)
        stat_dict[line]['std'] = np.std(rating_list)
        stat_dict[line]['max'] = np.amax(rating_list)
        stat_dict[line]['min'] = np.amin(rating_list)
        
    return stat_dict
        

def get_stats():
    #calls the functions to get the stats on the dicts and list
    return get_maths(user_dict), get_maths(movie_dict)
  
def show_stats_wk1(user_stats, movie_stats):
    print("Week 1 Question 1: Total number of Users is", len(user_stats))
    print("Week 1 Question 1: Total number of Movies is", len(movie_stats))
    print("Week 1 Question 1: Total number of Ratings is", len(rate_list))
    
    dens = (dat_len/(len(user_stats)*len(movie_stats)))*100
    #print ("Question 2: Density of full matrix is",np.round(dens,2))
    print("Week 1 Question 2: Density of full matrix is %.2f" % dens)
    
    total_user_mean = []
    total_user_median = []
    total_user_max = [] 
    total_user_min = [] 
    total_user_std = []
    
    for line in user_stats:
        total_user_mean.append(user_stats[line]['mean'])
        total_user_median.append(user_stats[line]['median'])     
        total_user_max.append(user_stats[line]['max'])     
        total_user_min.append(user_stats[line]['min'])     
        total_user_std.append(user_stats[line]['std'])     

    print("Week 1 Question 3: Average (per user) of mean {} median {} max {} min {} std {}" 
          .format(np.round(np.mean(total_user_mean),2), np.round(np.mean(total_user_median),2), np.round(np.mean(total_user_max),2),
           np.round(np.mean(total_user_min),2), np.round(np.mean(total_user_std),2)) )

    total_movie_mean = []
    total_movie_median = []
    total_movie_max = [] 
    total_movie_min = [] 
    total_movie_std = []
    
    for line in movie_stats:
        total_movie_mean.append(movie_stats[line]['mean'])
        total_movie_median.append(movie_stats[line]['median'])     
        total_movie_max.append(movie_stats[line]['max'])     
        total_movie_min.append(movie_stats[line]['min'])     
        total_movie_std.append(movie_stats[line]['std'])     

    print("Week 1 Question 4: Average (per movie) of mean {} median {} max {} min {} std {}" 
          .format(np.round(np.mean(total_movie_mean),2), np.round(np.mean(total_movie_median),2), np.round(np.mean(total_movie_max),2),
           np.round(np.mean(total_movie_min),2), np.round(np.mean(total_movie_std),2)) )

    print("Week 1 Question 5: Total number of Ratings is", Counter(rate_list))
    
    
def csv_output(user_stats, movie_stats):  
    user_stats_csv = open('user_stats.csv', 'w')  
    movie_stats_csv = open('movie_stats.csv', 'w')    
      
    writer = csv.writer(user_stats_csv)
    for key, value in user_stats.items():
        writer.writerow([key, value])
    
    writer = csv.writer(movie_stats_csv)
    for key, value in movie_stats.items():
        writer.writerow([key, value])
    user_stats_csv.close()
    movie_stats_csv.close()

#This is the optimal mean method. Rather than parsing the entire movie dict 
#and removing the user from the calculation it pops that user from the dict. Much
#faster than parsing the entire list
def mean_item_rating(user_id, movie_id):
    #create a dict variable from the current movie
    dict_tmp = (movie_dict[movie_id])
    #make a copy so we don't modify the original dict. I was having an issue
    #here where not making a copy seemed to remove ALL entries from the dict
    dict_tmp_cpy = dict_tmp.copy()    
    #pop out the user we don't want when calculating the average
    dict_tmp_cpy.pop(user_id)
    val_list = list(dict_tmp_cpy.values())
    #quickest way to check if list is empty
    #if the list is empty it means the current reviewer is the ONLY reviewer
    #and we want to discard the predction
    if val_list:
        return np.round(np.mean(val_list),3)
    else:
        return float('nan')


#This mean method was the original attempt. I modified it to the function above
#which improved the run time by 50%
def mean_item_rating_slow(user_id, movie_id):
    rating_list = []
    if len(movie_dict[movie_id]) == 1:
       #If there is only one movie in the list then we discard this prediction
       #as it must be the current user that has made it
       return float('nan')       
    else:
        #Otherwise append the rating to list to be averaged at the end
        for val in movie_dict[movie_id].items():
            if val[0] != user_id: 
                rating_list.append(val[1])         
        #round the average to 2dp and return 
        return np.round(np.mean(rating_list),3)    
   
def show_rmse_info_ave():
    #create an empty matrix 
    rmse_matrix_size = (dat_len,5)
    rmse_matrix = np.empty(rmse_matrix_size)
    i = 0
    #populate matrix with dict values
    for user in user_dict:
        for movie in user_dict[user]:            
            rmse_matrix[i,0] = user
            rmse_matrix[i,1] = movie
            rmse_matrix[i,2] = user_dict[user][movie]
            rmse_matrix[i,3] = pred_dict[user][movie]
            rmse_matrix[i,4] = rmse_dict[user][movie]
            i += 1
    #convert matrix to dataframe for easier conversion to CSV and parsing of columns    
    rmse_df = pd.DataFrame(rmse_matrix, columns=['user', 'movie', 'actual', 'predicted', 'RMSE'])      
    rmse_df.to_csv("rmse_average.csv", index=False) 
    print("CSV output")
    num_good_preds = dat_len - np.isnan(rmse_list_ave).sum()
    #no need to divide by 100k and mult by 100, just divide by 1000 to get percentage
    print("Coverage metric is for average method is {}%".format(num_good_preds/float(1000)))
    #to get RMSE ensure to drop Null values
    print("RMSE method average: {}".format(np.round(np.sqrt(np.nanmean(rmse_list_ave)),3)))


def show_rmse_info_neb(name):
    #create an empty matrix 
    rmse_matrix_size = (dat_len,5)
    rmse_matrix = np.empty(rmse_matrix_size)
    i = 0
    #populate matrix with dict values
    for user in user_dict:
        for movie in user_dict[user]:            
            rmse_matrix[i,0] = user
            rmse_matrix[i,1] = movie
            rmse_matrix[i,2] = user_dict[user][movie]
            rmse_matrix[i,3] = pred_dict_neb[user][movie]
            rmse_matrix[i,4] = rmse_dict_neb[user][movie]
            i += 1
    #convert matrix to dataframe for easier conversion to CSV and parsing of columns    
    rmse_df = pd.DataFrame(rmse_matrix, columns=['user', 'movie', 'actual', 'predicted(neighbour)', 'RMSE(neighbour)'])      
    rmse_df.to_csv("rmse_%s.csv" % name, index=False) 
    print("CSV output")
    num_good_preds = dat_len - np.isnan(rmse_list_neb).sum()
    coverage = (num_good_preds/float(dat_len))*100
    rmse = np.round( np.sqrt(np.nanmean(rmse_list_neb)   ),3)
    print("Neighbour size, coverage, RMSE {}, {}, {}".format(name, coverage, rmse))
    stats_list.append([name, coverage, rmse])
    show_time()


def show_rmse_info_cos():
    #create an empty matrix 
    rmse_matrix_size = (dat_len,5)
    rmse_matrix = np.empty(rmse_matrix_size)
    i = 0
    #populate matrix with dict values
    for user in user_dict:
        for movie in user_dict[user]:            
            rmse_matrix[i,0] = user
            rmse_matrix[i,1] = movie
            rmse_matrix[i,2] = user_dict[user][movie]
            rmse_matrix[i,3] = pred_dict_cos[user][movie]
            rmse_matrix[i,4] = rmse_dict_cos[user][movie]
            i += 1
    #convert matrix to dataframe for easier conversion to CSV and parsing of columns    
    rmse_df = pd.DataFrame(rmse_matrix, columns=['user', 'movie', 'actual', 'predicted(cos)', 'RMSE(cos)'])      
    rmse_df.to_csv("rmse_cos.csv", index=False) 
    print("CSV output")
    num_good_preds = dat_len - np.isnan(rmse_list_cos).sum()
    coverage = (num_good_preds/float(dat_len))*100
    rmse = np.round( np.sqrt(np.nanmean(rmse_list_cos)   ),3)
    print("Coverage, RMSE {}, {}".format(coverage, rmse))
    stats_list.append([coverage, rmse])
    show_time()

def show_rmse_info_res():
    #create an empty matrix 
    rmse_matrix_size = (dat_len,5)
    rmse_matrix = np.empty(rmse_matrix_size)
    i = 0
    #populate matrix with dict values
    for user in user_dict:
        for movie in user_dict[user]:            
            rmse_matrix[i,0] = user
            rmse_matrix[i,1] = movie
            rmse_matrix[i,2] = user_dict[user][movie]
            rmse_matrix[i,3] = pred_dict_res[user][movie]
            rmse_matrix[i,4] = rmse_dict_res[user][movie]
            i += 1
    #convert matrix to dataframe for easier conversion to CSV and parsing of columns    
    rmse_df = pd.DataFrame(rmse_matrix, columns=['user', 'movie', 'actual', 'predicted(res)', 'RMSE(res)'])      
    rmse_df.to_csv("rmse_res.csv", index=False) 
    print("CSV output")
    num_good_preds = dat_len - np.isnan(rmse_list_res).sum()
    coverage = (num_good_preds/float(dat_len))*100
    rmse = np.round( np.sqrt(np.nanmean(rmse_list_res)   ),3)
    print("Coverage, RMSE {}, {}".format(coverage, rmse))
    stats_list.append([coverage, rmse])
    show_time()



def create_stats_csv():

    stats_df = pd.DataFrame(stats_list, columns=['n', 'cov', 'RMSE']) 
    stats_df.plot(x='n', y='cov', style='o')
    stats_df.plot(x='n', y='RMSE', style='o')
    stats_df.to_csv("stats.csv", index=False) 
    

def create_pred_dict(method):
    #For each movie give the users with the rating they gave
    print("Predicting each movie based on average...")
    show_time()
    for user in user_dict:
        for movie in user_dict[user]:
            #I created 2 methods to calculate the mean
            #One was found to be 40 slower 
            if method == 'slow':
                #This method searches the entire dict and then calculates the mean
                #without the current user
                pred_dict[user][movie] = mean_item_rating_slow(user, movie)
            else:   
                #A faster method is to "pop" the user off and then calculate the mean
                pred_dict[user][movie] = mean_item_rating(user, movie)

def create_pred_dict_neb(neb_size):
    #For each movie give the users with the rating they gave
    print("Creating predictions based on neighbourhood size: {}...".format(neb_size))
    show_time()    
    for user in user_dict:
        for movie in user_dict[user]:
            #print("Getting prediction for user: {} for movie {}".format(user, movie))
            pred_dict_neb[user][movie] = get_pred_neb(user, movie, neb_size)

def create_pred_dict_cos():
    #For each movie give the users with the rating they gave
    print("Creating predictions based on cos method...")
    show_time()    
    for user in user_dict:
        for movie in user_dict[user]:
            #print("Getting prediction for user: {} for movie {}".format(user, movie))
            pred_dict_cos[user][movie] = get_pred_cos(user, movie)

def create_pred_dict_res():
    #For each movie give the users with the rating they gave
    print("Creating predictions based on res method...")
    show_time()    
    for user in user_dict:
        for movie in user_dict[user]:
            #print("Getting prediction for user: {} for movie {}".format(user, movie))
            pred_dict_res[user][movie] = get_pred_res(user, movie)

def create_ave_dict():
    #For each movie give the users with the rating they gave
    print("Creating ave dict based on res method...")
    for user in user_dict:
        user_vals = []        
        for movie in user_dict[user]:
            #print("Getting prediction for user: {} for movie {}".format(user, movie))
            user_vals.append((user_dict[user][movie]))
        user_ave_dict[user] = np.round(np.mean(user_vals),3)
        
def create_rmse_dict(method='neb'):
    del rmse_list_neb[:]
    del rmse_list_ave[:]
    del rmse_list_cos[:]

    show_time()
    if method == 'neb':
        print("Calculating RMSE values neighbour method...")
        for user in user_dict.keys():
            for movie in user_dict[user]:
                #for neighborhood
                #abs_val_neb = abs(pred_dict_neb[user][movie] - user_dict[user][movie])
                sq_val = ((pred_dict_neb[user][movie] - user_dict[user][movie]) ** 2)
                rmse_dict_neb[user][movie] = sq_val
                rmse_list_neb.append(sq_val)       
    elif method == 'cos':
        print("Calculating RMSE values cos method...")
        for user in user_dict.keys():
            for movie in user_dict[user]:
                #for neighborhood
                #abs_val_cose = abs(pred_dict_cose[user][movie] - user_dict[user][movie])
                sq_val = ((pred_dict_cos[user][movie] - user_dict[user][movie]) ** 2)
                rmse_dict_cos[user][movie] = sq_val
                rmse_list_cos.append(sq_val)
    elif method == 'res':
        print("Calculating RMSE values res method...")
        for user in user_dict.keys():
            for movie in user_dict[user]:
                #for neighborhood
                #abs_val_cose = abs(pred_dict_cose[user][movie] - user_dict[user][movie])
                sq_val = ((pred_dict_res[user][movie] - user_dict[user][movie]) ** 2)
                rmse_dict_res[user][movie] = sq_val
                rmse_list_res.append(sq_val)
    else:
        print("Calculating RMSE values average method...")
        for user in user_dict.keys():
            for movie in user_dict[user]:
                #instead of calculating the square and then square rooting I just get 
                #the absolute value of the error which is the same
                #rmse_dict[user][movie] = (pred_dict[user][movie] - user_dict[user][movie]) ** 2            
                sq_val = (pred_dict[user][movie] - user_dict[user][movie]) ** 2
                rmse_dict[user][movie] = sq_val
                #let's put all the rmse values in a list too, we can use it to get the mean
                #at the end. Saves doing it twice
                rmse_list_ave.append(sq_val)        

def rmse_timer(method):  
    #test the performance of the prediciton calculator
    time_taken = []
    num_loops = 10
    print("Testing performance...")
    
    if method == 'average':
        time_taken = []
        print("Making predictions using average method...")
        for i in range(num_loops):    
            t0 = time.time()
            #create_rmse_matrix_df()
            create_pred_dict('fast')
            create_rmse_dict()
            show_rmse_info_ave()
            t1 = time.time()
            time_taken.append(t1-t0)
            print("Loop {} time taken: {}s".format(i, np.round(t1-t0,2)))
        
        print("Average Time taken for fast method: {}s".format(np.round(np.mean(time_taken),2)))

    if method == 'neighbour':
        time_taken = []
        print("Making predictions using neighbour method...")
        for i in range(num_loops):    
            t0 = time.time()
            create_pred_dict_neb(5)
            create_rmse_dict()
            show_rmse_info_neb(5)
            t1 = time.time()
            time_taken.append(t1-t0)
            print("Loop {} time taken: {}s".format(i, np.round(t1-t0,2)))
        
        print("Average Time taken for neighbour method: {}s".format(np.round(np.mean(time_taken),2)))


def calc_sim(user1, user2):
    dict1 = user_dict[user1]
    dict2 = user_dict[user2]   
    intersection = set(dict1.keys()) & set(dict2.keys())
    
    #need to make sure there are actually common ratings between 2 users
    if(len(intersection) == 0):
        return 0  
    else:
        sim = 0
        #for each movie they have in common square the difference in review scores
        for cur_key in intersection:
            sim += (math.pow((dict1[cur_key] - dict2[cur_key]),2))/float(len(intersection))        
    return 1 - (sim/float(16))
    

def calc_sim_cos(user1, user2):
    dict1 = user_dict[user1]
    dict2 = user_dict[user2]   
    del movie_list1[:]
    del movie_list2[:]
    #get the list of movies they both have in common
    intersection = set(dict1.keys()) & set(dict2.keys())
    #seperate denominator and numerator    
    numer = 0
    denom1 = 0
    denom2 = 0
    #first make sure there are actually movies in common
    if(len(intersection) == 0):
        return 0  
    else:
        for cur_key in intersection:
            numer +=   dict1[cur_key] * dict2[cur_key]   

        for movie in dict1.keys():            
            denom1 += math.pow((dict1[movie]),2)
        for movie in dict2.keys():    
            denom2 += math.pow((dict2[movie]),2)
    denom_full = np.sqrt(denom1 * denom2)  
   
    return numer / denom_full

def calc_sim_res(user1, user2):
    dict1 = user_dict[user1]
    dict2 = user_dict[user2]   
    del movie_list1[:]
    del movie_list2[:]
    #get the list of movies they both have in common
    intersection = set(dict1.keys()) & set(dict2.keys())
    #seperate denominator and numerator    
    numer = 0
    denom1 = 0
    denom2 = 0
    #first make sure there are actually movies in common
    if(len(intersection) == 0):
        return 0  
    else:
        for cur_key in intersection:
            numer +=   (dict1[cur_key] - user_ave_dict[user1]) * (dict2[cur_key] - user_ave_dict[user2])

        for movie in dict1.keys():            
            denom1 += math.pow((dict1[movie] - user_ave_dict[user1]),2)
        for movie in dict2.keys():    
            denom2 += math.pow((dict2[movie] - user_ave_dict[user2]),2)
    denom_full = np.sqrt(denom1) * np.sqrt(denom2)    
   
    return numer / denom_full

    
def create_sim_matrix(method='neb'):
    print("Creating similarity matrix...")
    # we want to populate a matrix based on the similarty score for each user pair
    show_time()    
    i = 0
    #for every user compare against each other
    for user1_sim in user_dict.keys():
        j = 0
        for user2_sim in user_dict.keys():
            if method == 'neb':
                sim_matrix[i,j] = calc_sim(user1_sim, user2_sim)
            elif method == 'cos':
                sim_matrix[i,j] = calc_sim_cos(user1_sim, user2_sim)
            else:
                #print("Doing RES sim")
                sim_matrix[i,j] = calc_sim_res(user1_sim, user2_sim)
            j += 1
        i += 1  
        
def get_pred_neb(this_user,movie,neb_size):
    sim_list = []
    #since the matrix starts at zero and the user starts at 1, weed need to
    #subtract one to get the correct row
    user_row = sim_matrix[this_user-1,:]
    #sort the row in descending order
    user_row_sort = np.argsort(user_row)
    #reverse it to get most simlilar on top
    user_row_sort_rev = user_row_sort[::-1]
    #add one to each as we have returned indices which are one less than the user number
    user_row_sort_rev_inc = user_row_sort_rev + 1
    #for each user in that ordered list
    for user in user_row_sort_rev_inc:
        #if the target movie is in their list then add them to a similarity list
        if movie in user_dict[user]:
            sim_list.append(user)
    #take off the top user as it will always be this_user              
    del sim_list[0]
    #slice the neighbourhood size that we want
    neb_sim_list = sim_list[:neb_size] 
    if len(neb_sim_list) <  neb_size:
        #len of sim list too small discarding rating for user
        return float('nan')
    
    numer = 0
    denom = 0    
    for user_sim in neb_sim_list:
        #pred is sum of weight by rating for user on this movie divided by sim of ratings
        weight = sim_matrix[user_sim-1,this_user-1]
        numer += weight * user_dict[user_sim][movie]
        denom += weight
       
    return numer/denom      

def get_pred_cos(this_user,movie):
    numer = 0
    denom = 0 
    for user_sim in user_dict.keys():
        #pred is sum of weight by rating for user on this movie divided by sim of ratings
        weight = sim_matrix[user_sim-1,this_user-1]
        
        #need to check that we are not using the user's movie in the comparison
        if movie in user_dict[user_sim] and user_sim != this_user:
            numer += weight * user_dict[user_sim][movie]
            denom += weight
    #if denom is 0 that means this use is the only user of the movie and we discard it            
    if denom != 0:
        return numer/denom      
    else:
        return float('nan')

def get_pred_res(this_user,movie):
    numer = 0
    denom = 0 
    user_ave = user_ave_dict[this_user]
    #average_user_rating     + sum (rating(other_user, this_item) - other_user_average) x weight
    for user_sim in user_dict.keys():
        #pred is sum of weight by rating for user on this movie divided by sim of ratings
        weight = sim_matrix[user_sim-1,this_user-1]
        
        #need to check that we are not using the user's movie in the comparison
        if movie in user_dict[user_sim] and user_sim != this_user:
            #print("user_sim ave {} weight {}".format( user_ave_dict[user_sim], weight))
            numer +=  (user_dict[user_sim][movie] - user_ave_dict[user_sim]) * weight
            denom += abs(weight)
    #if denom is 0 that means this use is the only user of the movie and we discard it            
    if denom != 0:
        return user_ave + numer/denom 
        #return denom     
    else:
        return float('nan')


        
def show_time():
    print("--- {} seconds ---".format(np.round(time.time() - start_time),1))


    