
# coding: utf-8

# #Judd's Playground
# It's not as fun as it sounds

# ##Common

# In[247]:

get_ipython().magic(u'matplotlib inline')

import os
import csv
import collections

import datetime
import math
from random import randrange

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn import metrics


print os.getcwd()


#define function to remove borders
#borrowed from: http://nbviewer.ipython.org/github/cs109/content/blob/master/lec_03_statistical_graphs.ipynb
def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    """
    Minimize chartjunk by stripping out unnecesasry plot borders and axis ticks
    
    The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
    """
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()


#function to create a file for submission, given a set of predicted labels
def write_submission(pred_labels):
  #read in sample_submission
  with open('sampleSubmission.csv','rb') as rfile:
    reader = csv.reader(rfile)
    sample_data = [i for i in reader] 

  #put in header line
  submission_data = [sample_data[0]]

  #zip the datetime with predicted labels
  for i in range(1,len(sample_data)):
    submission_data.append([sample_data[i][0],pred_labels[i-1]])
  
  submission_data = np.array(submission_data)

  #confirm shape
  print submission_data.shape

  #unique file name
  now=str(datetime.datetime.now())
  fname = "submission - "+now+".csv"
  print fname

  #write to file
  with open(fname, 'wb') as writefile:
    writer = csv.writer(writefile) #create csv writer
    writer.writerows(submission_data)

#function to calculate RMSLE; the inputs must be log scale
def RMSLE(pred, act):
  RMSLE = 0.0
  for i in range(len(pred)):
    RMSLE += (pred[i] - act[i])**2
  RMSLE = (RMSLE/len(pred))**0.5
  return RMSLE

# function to print a header text along with a divider line
def print_div_line(text):
    print text + '_'*(80-len(text))


# ##Data Load Functions
# In[345]:
# Declare constants for easier readability
# -- Original Features
DATETIME = 0
SEASON = 1
HOLIDAY = 2
WORKINGDAY = 3
WEATHER = 4
TEMP = 5
ATEMP = 6
HUMIDITY = 7
WINDSPEED = 8
# -- Labels
CASUAL = 0
REGISTERED = 1
COUNT = 2
# -- DateTime Features
YEAR = 9
MONTH = 10
DAY = 11
HOUR = 12
# -- Added Features
DAYOFWEEK = 13
#FIRSTDAY = 14
#HIGH_WINDSPEED = 14
#MA_TEMP = 15
WEEKDAY_HOUR_AVG = 14
#NON_WORKINGDAY_COUNT = 16
#WET_GROUNDS = 16
MA_CASUAL = 15
MA_REGISTERED = 16
MA_COUNT = 17

def load_csv(file_name):
  """
    Reads in csv file with specified name
    Returns raw data from file as a list
  """
  with open(file_name, 'rb') as csv_file:
    reader = csv.reader(csv_file)
    data = [i for i in reader]
  
  return data


def load_train():
  """
    Reads in train.csv
    Returns
      feature_data: Numpy array of feature data with appropriate data types
      label_data: Numpy array of lable data with appropriate data types
      feature_names: Names of the features
      label_names: Names of the labels
    
    Data matches original csv file.  No transformations to data or added features
    are made here.
  """
  data = load_csv('train.csv')
  
  feature_names = data[0][:-3]
  label_names = data[0][-3:]
  
  # Create typed dataset
  typed_list = []
  for row in data[1:]:
    row[DATETIME] = datetime.datetime.strptime(row[DATETIME],'%Y-%m-%d %H:%M:%S')
    row[SEASON] = int(row[SEASON])
    row[HOLIDAY] = int(row[HOLIDAY])
    row[WORKINGDAY] = int(row[WORKINGDAY])
    row[WEATHER] = int(row[WEATHER])
    row[TEMP] = float(row[TEMP])
    row[ATEMP] = float(row[ATEMP])
    row[HUMIDITY] = int(row[HUMIDITY])
    row[WINDSPEED] = float(row[WINDSPEED])
    row[9+CASUAL] = int(row[9+CASUAL])
    row[9+REGISTERED] = int(row[9+REGISTERED])
    row[9+COUNT] = int(row[9+COUNT])
    typed_list.append(row)
  
  typed_array = np.array(typed_list)
  feature_data = typed_array[:,:-3]
  label_data = typed_array[:,-3:]
  
  return feature_data, label_data, feature_names, label_names

def load_test():
  """
    Reads in test.csv
    Returns
      data: Numpy array of feature data with appropriate data types
    
    Data matches original csv file.  No transformations to data or added features 
    are made heare.
  """
  data = load_csv('test.csv')
  
  # Create typed dataset
  typed_list = []
  for row in data[1:]:
    row[DATETIME] = datetime.datetime.strptime(row[DATETIME],'%Y-%m-%d %H:%M:%S')
    row[SEASON] = int(row[SEASON])
    row[HOLIDAY] = int(row[HOLIDAY])
    row[WORKINGDAY] = int(row[WORKINGDAY])
    row[WEATHER] = int(row[WEATHER])
    row[TEMP] = float(row[TEMP])
    row[ATEMP] = float(row[ATEMP])
    row[HUMIDITY] = int(row[HUMIDITY])
    row[WINDSPEED] = float(row[WINDSPEED])
    typed_list.append(row)
 
  typed_array = np.array(typed_list)
  return typed_array[:,:]

def log_transform_array(data):
  log_data = np.zeros(data.shape)
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      log_data[i,j] = np.log(label_data[i,j]+1)
  
  return log_data

# ##Data Cleansing Functions
# In[343]:
def atemp_outlier_fix(data):
  """
  fix outliers for atemp
  they all fall on august 17, 2012
  """
  data_new = data
  for i in range(len(data)):
    if data[i][TEMP]>20 and data[i][ATEMP]==12.12:
      data_new[i][ATEMP] = data[i][TEMP] + 3

  return data_new

def load_data():
    train_csv_data, label_data, csv_feature_names, label_names = load_train()
    train_csv_data = atemp_outlier_fix(train_csv_data)

    log_label_data = log_transform_array(label_data)
    test_csv_data = load_test()
    print 'Feature names: ', csv_feature_names
    print 'Train shape: ', train_csv_data.shape
    print 'Label names: ', label_names
    print 'Label shape: ', label_data.shape
    print 'Log Label shape: ', log_label_data.shape
    print 'Test shape: ', test_csv_data.shape

    return train_csv_data, label_data, csv_feature_names, label_names, log_label_data, test_csv_data

# ##Feature Engineering Functions
# Remember to add constants for column indexes of new features

# In[335]:

def add_separate_to_array(array_data): 
  new_columns = np.zeros((len(array_data), 4))
  for row_i in range(len(array_data)):
    new_columns[row_i,0] = array_data[row_i,DATETIME].year
    new_columns[row_i,1] = array_data[row_i,DATETIME].month
    new_columns[row_i,2] = array_data[row_i,DATETIME].day
    new_columns[row_i,3] = array_data[row_i,DATETIME].hour
  array_data = np.hstack((array_data, new_columns))
  
  return array_data
  
  
def separate_datetime(train_data, test_data, feature_names):
  feature_names = np.hstack((feature_names, np.array(['year', 'month', 'day', 'hour'])))
  train_new = add_separate_to_array(train_data)
  test_new = add_separate_to_array(test_data)
  
  return train_new, test_new, feature_names

def add_dayofweek_to_array(array_data):
  new_column = np.zeros((len(array_data),1))
  for row_i in range(len(array_data)):
    new_column[row_i,0] = array_data[row_i,DATETIME].weekday()
  array_data = np.hstack((array_data, new_column))
  
  return array_data

def add_dayofweek(train_data, test_data, feature_names):
  feature_names = np.hstack((feature_names, np.array(['day_of_week'])))
  
  train_new = add_dayofweek_to_array(train_data)
  test_new = add_dayofweek_to_array(test_data)
      
  return train_new, test_new, feature_names

def get_expected_prev_hour(current_hour, current_day, prev_hour_x):
  #print 'Current hour: ', current_hour, ' Current day: ', current_day, 'Prev x: ', prev_hour_x
  if current_hour-prev_hour_x < 0:
    if current_day == 1:
      expected_prev_hour = -1
    else:
      expected_prev_hour = 24+(current_hour-prev_hour_x)
  else:
    expected_prev_hour = current_hour - prev_hour_x
  return expected_prev_hour
  
def add_ma_x_to_array(label_data, array_data, x):
  new_columns = np.zeros((len(array_data),3))
  
  for row_i in range(len(array_data)):
    #print_div_line('Row'+str(row_i))
    ma_values = [] 
    missing_hours = 0
    
    for hour_i in range(x):
      #print 'Hour i: ', hour_i
      expected_prev_hour = get_expected_prev_hour(array_data[row_i,HOUR],
                                    array_data[row_i,DAY],
                                    hour_i+1)
      #print 'expected: ', expected_prev_hour
      actual_i = row_i - (hour_i+1-missing_hours)
      #print 'actual: ', actual_i
      if actual_i >= 0 and array_data[actual_i,HOUR] == expected_prev_hour:
        new_row = [label_data[actual_i,CASUAL],
                   label_data[actual_i,REGISTERED],
                   label_data[actual_i,COUNT]]
        ma_values.append(new_row)
      else:
        missing_hours += 1
    
    ma_values = np.asarray(ma_values)
    #print ma_values
    ma_mean = np.mean(ma_values, axis=0)
    #print 'Mean: ', ma_mean
    
    if ma_values.shape[0] == 0:
      new_columns[row_i,CASUAL] = label_data[row_i,CASUAL]
      new_columns[row_i,REGISTERED] = label_data[row_i,REGISTERED]
      new_columns[row_i,COUNT] = label_data[row_i,COUNT]
    else:
      new_columns[row_i,CASUAL] = ma_mean[CASUAL]
      new_columns[row_i,REGISTERED] = ma_mean[REGISTERED]
      new_columns[row_i,COUNT] = ma_mean[COUNT]
  
  array_data = np.hstack((array_data, new_columns))
  
  return array_data
  
def add_ma_x(train_data, test_data, feature_names, train_labels, test_labels, x):
    #print 'add_ma_x param check'
    #print 'train data: ', train_data.shape, ' train labels: ', train_labels.shape
    #print 'test data: ', test_data.shape, ' test labels: ', test_labels.shape
    #print 'ma x: ', x
    #print
    
    feature_names = np.hstack((feature_names, np.array(['ma_casual'+str(x),
                                                      'ma_registered'+str(x),
                                                      'ma_count'+str(x)])))
    train_new = add_ma_x_to_array(train_labels, train_data, x)
    test_new = add_ma_x_to_array(test_labels, test_data, x)
    
    return train_new, test_new, feature_names

def add_firstday_to_array(array_data, first_day):
    new_column = np.zeros((len(array_data),1))
    for row_i in range(len(array_data)):
        if array_data[row_i,DAY] == first_day:
            new_column[row_i,0] = 1
    array_data = np.hstack((array_data, new_column))  
    return array_data

def add_firstday(train_data, test_data, feature_names):
    feature_names = np.hstack((feature_names, np.array(['first_day'])))
  
    train_new = add_firstday_to_array(train_data, 1)
    test_new = add_firstday_to_array(test_data,17)
      
    return train_new, test_new, feature_names

def add_high_windspeed_to_array(array_data, threshold):
    new_column = np.zeros((len(array_data),1))
    #new_coloumn = array_data[:,WINDSPEED] > threshold
    for row_i in range(len(array_data)):
        if array_data[row_i,WINDSPEED] > threshold:
            new_column[row_i,0] = 1
    #print np.sum(new_column, axis=0)
    array_data = np.hstack((array_data, new_column))  
    return array_data

def add_high_windspeed(train_data, test_data, feature_names, threshold):
    feature_names = np.hstack((feature_names, np.array(['high_windspeed'])))
  
    train_new = add_high_windspeed_to_array(train_data, threshold)
    test_new = add_high_windspeed_to_array(test_data, threshold)
      
    return train_new, test_new, feature_names

def add_ma_x_feature_to_array(array_data, x, feature_column):
    new_columns = np.zeros((len(array_data),1))
  
    for row_i in range(len(array_data)):
        ma_values = [] 
        missing_hours = 0

        for hour_i in range(x):
          expected_prev_hour = get_expected_prev_hour(array_data[row_i,HOUR],
                                        array_data[row_i,DAY],
                                        hour_i+1)
          actual_i = row_i - (hour_i+1-missing_hours)
          if actual_i >= 0 and array_data[actual_i,HOUR] == expected_prev_hour:
            new_row = [array_data[actual_i,feature_column]]                   
            ma_values.append(new_row)
          else:
            missing_hours += 1

        ma_values = np.asarray(ma_values)
        ma_mean = np.mean(ma_values, axis=0)

        if ma_values.shape[0] == 0:
          new_columns[row_i,0] = array_data[row_i,feature_column]
        else:
          new_columns[row_i,0] = ma_mean[0]

    array_data = np.hstack((array_data, new_columns))

    return array_data

def add_ma_x_feature(train_data, test_data, feature_names, x, feature_name, feature_column):   
    feature_names = np.hstack((feature_names, np.array(['ma_'+feature_name+str(x)])))
    train_new = add_ma_x_feature_to_array(train_data, x, feature_column)
    test_new = add_ma_x_feature_to_array(test_data, x, feature_column)
    
    return train_new, test_new, feature_names

def calc_weekday_hour_avg(data, labels, current_dow, current_hour, current_date, m_x):
    weekday_hour_avg = 0
    weekday_hour_labels = labels[(data[:,DAYOFWEEK] == current_dow) & 
                                 (data[:,HOUR] == current_hour) &
                                 (data[:,DATETIME] < current_date) &
                                 (data[:,DATETIME] >= data[:,DATETIME]-datetime.timedelta(weeks=m_x*4))][:,COUNT]
    if weekday_hour_labels.shape[0] > 0:
        weekday_hour_avg = weekday_hour_labels.mean()
    else:
        # If at beginning of data, no previous dates to get data from, just use data from that day
        # This should only occur at beginning of train data
        weekday_hour_labels = labels[(data[:,DAYOFWEEK] == current_dow) & 
                                 (data[:,HOUR] == current_hour) &
                                 (data[:,DATETIME] == current_date)][:,COUNT]
        weekday_hour_avg = weekday_hour_labels.mean()
        
    return weekday_hour_avg
            
def add_weekday_hour_avg_to_array(data, train_data, train_labels, m_x):
    new_column = np.zeros((len(data),1))
    for row_i in range(len(data)):
        new_column[row_i,0] = calc_weekday_hour_avg(train_data, train_labels, 
                                                    data[row_i,DAYOFWEEK], data[row_i,HOUR], data[row_i,DATETIME], m_x)
    data = np.hstack((data, new_column))  
    return data

def add_weekday_hour_avg(train_data, test_data, feature_names, train_labels, m_x):
    feature_names = np.hstack((feature_names, np.array(['weekday_hour_avg'])))
    
    #weekday_hour_avg_matrix = np.zeros((7,24))
    #for weekday in range(7):
    #    for hour in range(24):
    #        weekday_hour_labels = train_labels[(train_data[:,DAYOFWEEK] == weekday) & 
    #                                           (train_data[:,HOUR] == hour)][:,COUNT]
    #        weekday_hour_avg_matrix[weekday,hour] = weekday_hour_labels.mean()
    
    train_new = add_weekday_hour_avg_to_array(train_data, train_data, train_labels, m_x)
    test_new = add_weekday_hour_avg_to_array(test_data, train_data, train_labels, m_x)
    
    return train_new, test_new, feature_names


def remove_features(feature_list, feature_names, train_data, dev_data, test_data):
  feature_names = np.delete(feature_names, feature_list)
  train_data = np.delete(train_data, feature_list, axis=1)
  dev_data = np.delete(dev_data, feature_list, axis=1)
  test_data = np.delete(test_data, feature_list, axis=1)
  
  return feature_names, train_data, dev_data, test_data


# In[250]:

def build_day_dict(day_dict, feature, data, cond = 0):
  """
  build dictionary that counts number of consecutive days with same value of feature. 
  for example, if two consecutive days both have values workingday = 1, day_dict would have a 2 for each of those days
  """
  d_dict = {}
  day_arr, feature_arr = [], []
  for n in range(len(data)):
    #if datetime.datetime(int(data[n][YEAR]), int(data[n][MONTH]), int(data[n][DAY])) not in d_dict:
    if data[n,DATETIME].date() not in d_dict:
      d_dict[data[n,DATETIME].date()] = data[n,feature]

  ordered_day_dict = collections.OrderedDict(sorted(d_dict.items()))
  
  for key, value in ordered_day_dict.iteritems():
    feature_arr.append(value)
    day_arr.append(str(key))
  
  cond = 0
  length = []
  count = 0
  for i in range(len(feature_arr)):
      if feature_arr[i] == cond:
          count += 1
      elif count > 1:
          length.append(count)
          length.append(0)
          count = 0
      else:
          length.append(0)

      if i == len(feature_arr)-1 and count > 0:
          length.append(count)
  
  p = []
  for l in length:
    if l == 0:
      p.append(0)
    else:
      for i in range(l):
        p.append(l)
    
  for d, c in zip(day_arr, p):
    day_dict[d] = c
  
  return day_dict

def add_non_workday_count_to_array(array_data, day_dict):
    new_column = np.zeros((len(array_data),1))
    for row_i in range(len(array_data)):
        new_column[row_i,0] = day_dict[str(array_data[i,DATETIME].date())]
    array_data = np.hstack((array_data, new_column))  
    return array_data

def add_non_workday_count(train_data, test_data, feature_names):
    feature_names = np.hstack((feature_names, np.array(['non_workday_count'])))
    
    day_dict = build_day_dict({}, WORKINGDAY, np.vstack((train_data, test_data)))
    
    train_new = add_non_workday_count_to_array(train_data, day_dict)
    test_new = add_non_workday_count_to_array(test_data, day_dict)
    
    return train_new, test_new, feature_names


# In[336]:

def build_past_period_dict(p_dict, period, feature, threshold, data):
    """
    build dictionary that counts number of occurances of a certain value in the past x hours.
    for example, count number of hours with weather > 2 in the last 3 hours.
    you should feed combined (train,dev,test) to data
    for wet-grounds, set period to 3, feature to weather, threshold to 2
    """
    return_dict = {}

    for n in range(len(data)):
        if data[n,DATETIME] not in p_dict:
            p_dict[data[n,DATETIME]] = data[n,feature]

    ordered_dict = collections.OrderedDict(sorted(p_dict.items()))

    for key, value in ordered_dict.iteritems():
        date_list = [key - datetime.timedelta(hours = h_i+1) for h_i in range(period)]

        count = 0
        for d in date_list:
            if d in ordered_dict and ordered_dict[d] > threshold:
                count += 1

        return_dict[key] = count

    return return_dict

def add_wet_grounds_to_array(array_data, period_dict):
    new_column = np.zeros((len(array_data),1))
    for row_i in range(len(array_data)):
        new_column[row_i,0] = period_dict[array_data[i,DATETIME]]
    array_data = np.hstack((array_data, new_column))  
    return array_data

def add_wet_grounds(train_data, test_data, feature_names, x):
    feature_names = np.hstack((feature_names, np.array(['wet_grounds'])))
    
    period_dict = build_past_period_dict({}, x, WEATHER, 2, np.vstack((train_data, test_data)))

    train_new = add_wet_grounds_to_array(train_data, period_dict)
    test_new = add_wet_grounds_to_array(train_data, period_dict)
    
    return train_new, test_new, feature_names




# ##Train/Dev Split Functions

# In[337]:

def split_monthly(data, labels, log_labels, day_holdout):
  """
    For each month in original dataset, splits data into test and dev
      test: Data that comes before 20th - day_holdout DAY in month
      dev: Data that comes on or after 20th - day_holdout DAY in month
  """
  
  train_list, train_label_list, train_log_label_list = [],[],[]
  dev_list, dev_label_list, dev_log_label_list = [],[],[]
  for i in range(len(data)):
    if data[i][DAY] < 20-day_holdout:
      train_list.append(data[i])
      train_label_list.append(labels[i])
      train_log_label_list.append(log_labels[i])
    else:
      dev_list.append(data[i])
      dev_label_list.append(labels[i])
      dev_log_label_list.append(log_labels[i])
  
  train_data = np.array(train_list)
  train_labels = np.array(train_label_list)
  train_log_labels = np.array(train_log_label_list)
  
  dev_data = np.array(dev_list)
  dev_labels = np.array(dev_label_list)
  dev_log_labels = np.array(dev_log_label_list)
  
  return train_data, train_labels, train_log_labels, dev_data, dev_labels, dev_log_labels
    
def split_monthly_random(data, labels, log_labels, max_day_holdout, min_day_holdout):
    """
    For each month in original train, splits data into test and dev
    test: data that comes before random of the max holdout day in month
    dev: data that comes on or after day of random max holdout in month
    """
    train_list, train_label_list, train_log_label_list = [],[],[]
    dev_list, dev_label_list, dev_log_label_list = [],[],[]
    
    #print 'Data shape: ', data.shape
    
    split_days = [20 - randrange(max_day_holdout) - min_day_holdout for m in range(1,13)]
    #print split_days
    
    for i in range(len(data)):
        if data[i][DAY] < split_days[int(data[i][MONTH])-1]:
            train_list.append(data[i])
            train_label_list.append(labels[i])
            train_log_label_list.append(log_labels[i])
        else:
            dev_list.append(data[i])
            dev_label_list.append(labels[i])
            dev_log_label_list.append(log_labels[i])
    
    train_data = np.array(train_list)
    #print train_data
    train_labels = np.array(train_label_list)
    train_log_labels = np.array(train_log_label_list)
  
    dev_data = np.array(dev_list)
    #print dev_data
    dev_labels = np.array(dev_label_list)
    dev_log_labels = np.array(dev_log_label_list)
  
    return train_data, train_labels, train_log_labels, dev_data, dev_labels, dev_log_labels
    
#train_data, test_data, feature_names = separate_datetime(train_csv_data, 
#                                                         test_csv_data, 
#                                                         csv_feature_names)
#split_monthly_random(train_data, label_data, label_data, 5, 2)
    


# ##Model: Basic Random Forest Regressor
# This model does a random forest regression against the original data with out any transformations.
# 
# The datetime column is removed from this analysis, as the separate date value columns are used.
# 
# ###Best paramters: 
#        RandomForestRegressor(bootstrap=True, compute_importances=None,
#        criterion='mse', max_depth=12, max_features='auto',
#        max_leaf_nodes=None, min_density=None, min_samples_leaf=1.5,
#        min_samples_split=2, n_estimators=1000, n_jobs=1,
#        oob_score=False, random_state=None, verbose=0)

# In[338]:



def print_feature_importances(feature_names, feature_importances):
  print "Feature Importances:"
  for name, importance in zip(feature_names, feature_importances):
    print "%10s\t%6.4f"%(name, importance)

def rfr_dev_check(train_data, train_labels, dev_data, dev_labels, n_est):
  rfr = RandomForestRegressor(n_estimators=n_est)
                              #, bootstrap=True, compute_importances=None,
                              #criterion='mse', max_depth=12, max_features='auto', max_leaf_nodes=None,
                              #min_density=None, min_samples_leaf=1.5,
                              #min_samples_split=2, n_jobs=1, oob_score=False, random_state=None,
                              #verbose=0)
  rfr.fit(train_data, train_labels)
  pred_labels = rfr.predict(dev_data)

  print "%7.5f\t%7.5f\t%7.5f" % (round(RMSLE(pred_labels[:,CASUAL], dev_labels[:,CASUAL]),3), 
                                 round(RMSLE(pred_labels[:,REGISTERED], dev_labels[:,REGISTERED]),3),
                                 round(RMSLE(pred_labels[:,COUNT], dev_labels[:,COUNT]),3))
  print
  print_feature_importances(feature_names, rfr.feature_importances_)

def rfr_dev_predict(train_data, train_labels, dev_data, dev_labels, n_est):
    rfr = RandomForestRegressor(n_estimators=n_est)
    rfr.fit(train_data, train_labels)
    pred_labels = rfr.predict(dev_data)
  
    #print_feature_importances(feature_names, rfr.feature_importances_)
  
    return RMSLE(pred_labels[:,CASUAL], dev_labels[:,CASUAL]), RMSLE(pred_labels[:,REGISTERED], dev_labels[:,REGISTERED]), RMSLE(pred_labels[:,COUNT], dev_labels[:,COUNT])

    
def rfr_test_predict(combined_data, combined_labels, test_data, n_est):
    rfr = RandomForestRegressor(n_estimators=n_est)
    rfr.fit(combined_data, combined_labels)
    pred_labels = rfr.predict(test_data)
    
    #print 'OOB Score: ', rfr.oob_score_
    #print_feature_importances(feature_names, rfr.feature_importances_  
    return pred_labels
  
def rfr_submit(pred_log_labels):
  count_pred = np.exp(pred_log_labels[:,2]) - 1
  print "%d predicted count values <= 0" % len(count_pred[count_pred<0])

  for i in range(len(count_pred)):
    if count_pred[i] < 0:
      count_pred[i] = 0

  write_submission(count_pred)


# In[339]:

# Evaluate against dev
def evaluate_dev():
    n = 3
    iter_array = np.zeros((n,3))
    for i in range(n):
        # Add DATE PARTS
        combo_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

        # Add DAY OF WEEK
        combo_data, test_data, feature_names = add_dayofweek(combo_data, test_data, feature_names)

        # Add HIGH WINDSPEED
        #combo_data, test_data, feature_names = add_high_windspeed(combo_data, test_data, feature_names, 35)

        # Add WEEKDAY-HOUR-AVG
        combo_data, test_data, feature_names = add_weekday_hour_avg(combo_data, test_data, feature_names, label_data, 3)

        # Add NON-WORKINGDAY-COUNT
        #combo_data, test_data, feature_names = add_non_workday_count(combo_data, test_data, feature_names)

        # Add WET-GROUNDS
        #combo_data, test_data, feature_names = add_wet_grounds(combo_data, test_data, feature_names, 3)

        # Add MA
        combo_data, test_data, feature_names = add_ma_x(combo_data, test_data, feature_names, label_data, label_data, 2)

        # Split
        train_data, train_labels, train_log_labels, dev_data, dev_labels, dev_log_labels = split_monthly_random(combo_data, label_data, log_label_data, 10, 2)

        # Remove Columns
        feature_names, train_data, dev_data, test_data = remove_features([DATETIME],
                                                                         feature_names,
                                                                         train_data,
                                                                         dev_data,
                                                                         test_data)

        # Execute RFR
        iter_array[i,0], iter_array[i,1], iter_array[i,2] = rfr_dev_predict(train_data, train_log_labels,
                                                                            dev_data, dev_log_labels, 300)

    # Print results
    print iter_array
    iter_mean = np.mean(iter_array, axis=0)
    print_div_line('RFR')
    print 'Features: ', feature_names
    print
    print '`Means:\t', np.around(iter_mean, decimals=3), '`  '
    print '`Highs:\t', np.around(np.amax(iter_array, axis=0), decimals=3), '`  '
    print '`Lows:\t ', np.around(np.amin(iter_array, axis=0), decimals=3), '`  '


# ###Values History 3
# 
# ####2-MA +DoW +WK_H_AVG**3ma x 3 (MA before random split, Atemp Cleaned)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.453  0.283  0.281] `  
# `Highs:	[ 0.464  0.293  0.29 ] `  
# `Lows:	  [ 0.436  0.277  0.275] `
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C x 3 (MA before random split, Atemp Cleaned)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'non_workday_count' 'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.447  0.278  0.275] `  
# `Highs:	[ 0.455  0.279  0.278] `  
# `Lows:	  [ 0.434  0.278  0.271] ` 
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C x 3 (MA before random split)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'non_workday_count' 'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.45   0.274  0.275] `  
# `Highs:	[ 0.462  0.276  0.277] `  
# `Lows:	  [ 0.44   0.273  0.272] ` 
# 
# ####2-MA +DoW +H_WS +WK_H_AVG**3ma x 3 (MA before random split)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'high_windspeed'
#  'weekday_hour_avg' 'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.458  0.286  0.285] `  
# `Highs:	[ 0.461  0.289  0.287] `  
# `Lows:	  [ 0.455  0.281  0.28 ] ` 
# 
# ####2-MA +DoW +WK_H_AVG**3ma +WET x 3 (MA before random split)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'wet_grounds' 'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.458  0.281  0.278] `  
# `Highs:	[ 0.462  0.283  0.28 ] `  
# `Lows:	  [ 0.454  0.279  0.277] ` 
# 
# ####2-MA +HIGH_WS +WK_H_AVG +N_WKD_C +WET x 10 (MA before split, Random Split, Day of Week, High WS, WD_H_AVG, Non-Workingday-Count, Wet-grounds)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'non_workday_count' 'wet_grounds' 'ma_casual2' 'ma_registered2'
#  'ma_count2']
# 
# `Means:	[ 0.463  0.277  0.277] `  
# `Highs:	[ 0.48   0.291  0.291] `  
# `Lows:	 [ 0.438  0.265  0.266] ` 
# 
# ####2-MA +WK_H_AVG +N_WKD_C +WET x 10 (MA before split, Random Split, Day of Week, WD_H_AVG, Non-Workingday-Count, Wet-grounds)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'non_workday_count' 'wet_grounds' 'ma_casual2' 'ma_registered2'
#  'ma_count2']
# 
# `Means:	[ 0.465  0.275  0.275] `  
# `Highs:	[ 0.475  0.281  0.28 ] `  
# `Lows:	 [ 0.452  0.272  0.269] `  
# 
# ####2-MA +WET x 10 (MA before split, Random Split, Day of Week, Wet-grounds)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'wet_grounds'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.461  0.288  0.286] `  
# `Highs:	[ 0.475  0.299  0.295] `  
# `Lows:	 [ 0.453  0.281  0.277] ` 
# 
# ####2-MA +DoW +WK_H_AVG** x 3 (MA before split, Random Split, First with avg of only before dates w4-month MA)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.451  0.28   0.278] `  
# `Highs:	[ 0.465  0.284  0.285] `  
# `Lows:	[ 0.434  0.272  0.272] `
# 
# ####2-MA +DoW +WK_H_AVG** x 3 (MA before split, Random Split, First with avg of only before dates w2-month MA)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.464  0.288  0.287] `  
# `Highs:	[ 0.472  0.297  0.297] `  
# `Lows:	[ 0.455  0.283  0.282] ` 
# 
# ####2-MA +DoW +WK_H_AVG** x 3 (MA before split, Random Split, First with avg of only before dates w5-month MA)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.456  0.28   0.28 ] `  
# `Highs:	[ 0.465  0.283  0.283] `  
# `Lows:	[ 0.447  0.278  0.277] ` 
# 
# ####2-MA +DoW +WK_H_AVG** x 10 (MA before split, Random Split, First with avg of only before dates w3-month MA)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.454  0.277  0.275] `  
# `Highs:	[ 0.465  0.286  0.287] `  
# `Lows:	[ 0.439  0.27   0.268] ` 
# 
# ####2-MA +DoW +WK_H_AVG* x 10 (MA before split, Random Split, First with avg of only before dates)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.459  0.281  0.28 ] `  
# `Highs:	[ 0.481  0.298  0.297] `  
# `Lows:	[ 0.444  0.267  0.265] `  
# 
# ####2-MA +DoW +WK_H_AVG x 10 (MA before split, Random Split)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.464  0.277  0.276]`  
# `Highs:	[ 0.475  0.294  0.292]`  
# `Lows:	 [ 0.451  0.26   0.26 ]`
# 
# ####2-MA +N_WKD_C x 10 (MA before split, Random Split, Day of Week, Non-Workingday-Count)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'non_workday_count'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# `Means:	[ 0.461  0.286  0.284]`  
# `Highs:	[ 0.474  0.292  0.29 ]`  
# `Lows:	 [ 0.45   0.279  0.277]`
# 
# ####2-MA +HIGH_WS(35) +WK_H_AVG +N_WKD_C x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, WD_H_AVG, Non-Workingday-Count)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'high_windspeed'
#  'weekday_hour_avg' 'non_workday_count' 'ma_casual2' 'ma_registered2'
#  'ma_count2']
# 
# `Means:	[ 0.46   0.276  0.275]`
# 
# `Highs:	[ 0.472  0.285  0.284]`
# 
# `Lows:	[ 0.44   0.267  0.268]`
# 
# ####2-MA +HIGH_WS(35) +WK_H_AVG x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, WD_H_AVG)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity'
#  'windspeed' 'year' 'month' 'day' 'hour' 'day_of_week' 'high_windspeed'
#  'weekday_hour_avg' 'ma_casual2' 'ma_registered2' 'ma_count2']
# 
# Means:	[ 0.459  0.276  0.275]
# 
# Highs:	[ 0.478  0.285  0.285]
# 
# Lows:	[ 0.442  0.269  0.269]
# 
# ####2-MA +HIGH_WS(35) +WK_H_AVG -WS-DAY x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, WD_H_AVG, Remove WS,Day)
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'temp' 'atemp' 'humidity' 'year'
#  'month' 'hour' 'day_of_week' 'high_windspeed' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
#  
# Means:	[ 0.46060539  0.27563577  0.27484899]
# 
# Highs:	[ 0.4774874   0.28219482  0.28189837]
# 
# Lows:	[ 0.43583127  0.27068473  0.26703645]
# 
# ####2-MA +HIGH_WS(35) +WK_H_AVG -T-WS-DAY x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, WD_H_AVG, Remove Temp,WS,Day)####
# 
# Features:  ['season' 'holiday' 'workingday' 'weather' 'atemp' 'humidity' 'year'
#  'month' 'hour' 'day_of_week' 'high_windspeed' 'weekday_hour_avg'
#  'ma_casual2' 'ma_registered2' 'ma_count2']
#  
# Means:	[ 0.46638538  0.27648372  0.27724307]
# 
# Highs:	[ 0.48232348  0.29407207  0.29458454]
# 
# Lows:	[ 0.4380104   0.26567995  0.26592785]
# 

# ###Values History 2
# 
# ####2-MA +HIGH_WS(35) -T-WS-DAY x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, 2-MA-Temp, Remove Temp,WS,Day)
# 
# 0.285 High: 0.290 Low: 0.281
# 
# ####2-MA +HIGH_WS(35) -DAY x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, Remove Day)
# 
# 0.283 High: 0.288 Low: 0.278
# 
# ####2-MA +HIGH_WS(35) -WS-DAY x 10 (MA before split, Random Split, Day of Week, High windspeed > 35, Remove WS,Day)
# 
# 0.283 High: 0.291 Low: 0.270
# 
# ####2-MA +H_WS(35) +MA_TEMP-3 -DAY-WS x 5 (MA before split, Random Split, Day of Week, HWS > 35, 3-MA Temp, Remove WS,Day)
# 
# 0.471  0.287  0.287
# 
# ####2-MA +H_WS(35) +MA_TEMP-3 -DAY-WS-TEMP x 5 (MA before split, Random Split, Day of Week, HWS > 35, 3-MA Temp, Remove Temp,WS,Day)
# 
# 0.459  0.286  0.284
# 
# ####2-MA +H_WS(35) +MA_TEMP-5 -DAY-WS-TEMP x 5 (MA before split, Random Split, Day of Week, HWS > 35, 5-MA Temp, Remove Temp,WS,Day)
# 
# 0.461  0.287  0.286
# 
# ####2-MA +H_WS(35) +MA_TEMP-2 -TEMP-WS-DAY x 5 (MA before split, Random Split, Day of Week, HWS > 35, 2-MA Temp, Remove Temp,WS,Day)
# 
# 0.459  0.282  0.280
# 
# 0.463  0.286  0.285
# 
# 0.461  0.287  0.286
# 
# ####2-MA +H_WS(35) +MA_TEMP-2 -DAY-WS x 5 (MA before split, Random Split, Day of Week, HWS > 35, 2-MA Temp, Remove WS,Day)
# 
# 0.467  0.286  0.285
# 
# 0.463  0.287  0.285
# 
# ####2-MA +HIGH_WS(35) -DAY +WS x 5 (MA before split, Random Split, Day of Week, High windspeed > 35, Remove Day)
# 
# 0.461  0.284  0.284
# 
# ####2-MA +HIGH_WS(35) -DAY x 5 (MA before split, Random Split, Day of Week, High windspeed > 35, Remove Day, Windspeed)
# 
# 0.457  0.283  0.280 <b>Keep Remove Day</b>
# 
# 0.461  0.283  0.282
# 
# ####2-MA +HIGH_WINDSPEED(35) +WINDSPEED x 5 (MA before split, Random Split, Day of Week, High windspeed > 35)
# 
# 0.459  0.284  0.281 <b>Don't remove windspeed</b>
# 
# ####2-MA +HIGH_WINDSPEED(35) x 5 (MA before split, Random Split, Day of Week, High windspeed > 35, Remove Windspeed)
# 
# <b>0.466  0.282  0.281</b>  <b> Keep High Windspeed > 35</b>
# 
# 0.463  0.287  0.285
# 
# ####2-MA +HIGH_WINDSPEED(20) x 5 (MA before split, Random Split, Day of Week, High windspeed > 20, Remove Windspeed)
# 
# 0.460  0.291  0.288
# 
# ####2-MA +HIGH_WINDSPEED(30) x 5 (MA before split, Random Split, Day of Week, High windspeed > 30, Remove Windspeed)
# 
# 0.465  0.284  0.283
# 
# ####2-MA -WINDSPEED x 5 (MA before split, Random Split, Day of Week, Remove Windspeed)
# 
# 0.460  0.286  0.284
# 
# ####2-MA -DAY x 5 (MA before split, Random Split, Day of Week, Remove Day)
# 
# 0.459  0.286  0.283  <b>Possible Keep: Remove Day</b>
# 
# ####2-MA -MONTH x 5 (MA before split, Random Split, Day of Week, Remove Month)
# 
# 0.464  0.290  0.286
# 
# ####3-MA x 5 (MA before split, Random Split, Day of Week, No Removal)
# 
# 0.462  0.287  0.286
# 
# ####2-MA x 5 (MA before split, Random Split, Day of Week, No Removal)
# 
# <b>0.458  0.283  0.283</b>  <b>Keep 2-MA</b>
# 
# 0.461  0.287  0.285
# 
# ####Day of Week x 5 (Random Split, Day of Week, No MA, No Removal)
# 
# 0.556  0.334  0.343  <b>Keep Day of Week</b>
# 
# ####Raw x 5 (Random Split, No Day of Week, No MA, No Removal)
# 
# 0.579  0.362  0.371

# ###Values history
# 3-hour MA
# 
# 0.454  0.292  0.287
# 
# 2-hour MA
# 
# 0.456  0.289  0.284
# 
# 2-hour MA, remove month and day
# 
# 0.462  0.287  0.284
# 
# **2-hour MA, don't remove month and day
# 
# 0.451  0.284  0.280
# 
# 5-hour MA
# 
# 0.470  0.305  0.306
# 
# 2-hour MA on train and dev after split, remove month
# 
# 0.470  0.294  0.291
# 
# 2-hour MA split, remove day
# 
# 0.466  0.290 0.288
# 
# 2-hour MA split, remove month
# 
# 0.471  0.299  0.297
# 
# 2-hour MA split, remove month and day
# 
# 0.461  0.289  0.286
# 
# 2-hour MA split, add first day, remove month and day -- but need to do for dev first day!
# 
# 0.473  0.294  0.292
# 
# 2-hour MA split, no first day, remove month and day, no random split - just 17th dev start
# 
# 0.469 0.292  0.291
# 
# 2-hour MA split, WITH first day, remove month and day, no random split - just 17th dev start
# 
# 
# 

# ###Multi-pass testing grounds

# In[341]:

def evaluate_dev_multipass():
    n = 1
    iter_array = np.zeros((n,3))

    # Prepare baseline data
    combo_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

    # Prepare data with day of week feature
    combo_data, test_data, feature_names = add_dayofweek(combo_data, test_data, feature_names)

    # Prepare data with high windspeed feature
    #combo_data, test_data, feature_names = add_high_windspeed(combo_data, test_data, feature_names, 35)

    # Prepare data with weekday-hour avg feature
    combo_data, test_data, feature_names = add_weekday_hour_avg(combo_data, test_data, feature_names, label_data, 7)

    # Prepare data with non-workingday-count feature
    #combo_data, test_data, feature_names = add_non_workday_count(combo_data, test_data, feature_names)

    # Prepare data with wet-grounds feature
    #combo_data, test_data, feature_names = add_wet_grounds(combo_data, test_data, feature_names, 3)

    # Split train to train and dev
    train_data, train_labels, train_log_labels, dev_data, dev_labels, dev_log_labels = split_monthly(combo_data, label_data, log_label_data, 3)

    # Remove datetime,month column for the following analyses
    feature_names, train_data, dev_data, test_data = remove_features([DATETIME],
                                                                     feature_names,
                                                                     train_data,
                                                                     dev_data,
                                                                     test_data)

    # Execute RFR
    print_div_line('RFR, Initial')
    devhat_log_labels = rfr_test_predict(train_data, train_log_labels, dev_data, 300)
    rfr_dev_check(train_data, train_log_labels, dev_data, dev_log_labels, 300)
    print

    multi_pass = 1
    for mpass in range(multi_pass):
        # Reinit
        # Prepare baseline data
        combo_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

        # Prepare data with day of week feature
        combo_data, test_data, feature_names = add_dayofweek(combo_data, test_data, feature_names)

        # Prepare data with high windspeed feature
        #combo_data, test_data, feature_names = add_high_windspeed(combo_data, test_data, feature_names, 35)

        # Prepare data with weekday-hour avg feature
        combo_data, test_data, feature_names = add_weekday_hour_avg(combo_data, test_data, feature_names, label_data, 7)

        # Prepare data with non-workingday-count feature
        combo_data, test_data, feature_names = add_non_workday_count(combo_data, test_data, feature_names)

        # Prepare data with wet-grounds feature
        #combo_data, test_data, feature_names = add_wet_grounds(combo_data, test_data, feature_names, 3)

        # Split train to train and dev
        #train_data, train_labels, train_log_labels, dev_data, dev_labels, dev_log_labels = split_monthly(combo_data, label_data, log_label_data, 3)

        # Prepare data with moving average features
        devhat_labels = np.exp(devhat_log_labels) - 1
        train_data, dev_data, feature_names = add_ma_x(train_data, dev_data, feature_names,
                                                       train_labels, devhat_labels, 2)


        # Remove datetime,month column for the following analyses
        feature_names, train_data, dev_data, test_data = remove_features([DATETIME],
                                                                         feature_names,
                                                                         train_data,
                                                                         dev_data,
                                                                         test_data)

        # Execute RFR with predicted test_labels in the test_data set
        print_div_line('RFR, Pass ' + str(mpass))
        devhat_log_labels = rfr_test_predict(train_data, train_log_labels, dev_data, 300)
        rfr_dev_check(train_data, train_log_labels, dev_data, dev_log_labels, 300)
        print


# ###Value History
# 
# ####2-MA +DoW +WK_H_AVG**7ma (Static Split, Cleansed Atemp)
# 
# Pass 1: 0.521  0.311  0.316  Pass 2: 0.526  0.310  0.317
# 
# ####2-MA +DoW +WK_H_AVG**3ma (Static Split, Cleansed Atemp)
# 
# Pass 1: 0.522  0.311  0.317  Pass 2: 0.524  0.310  0.316
# 
# ####2-MA +DoW +H_WS(35) +WK_H_AVG**3ma (Static Split, Cleansed Atemp)
# 
# Pass 1: 0.520  0.313  0.318  Pass 2: 0.527  0.311  0.317
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C (Static Split, Cleansed Atemp)
# 
# Pass 1: 0.522  0.312  0.318  Pass 2: 0.515  0.305  0.314
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C -MONTH (Static Split, Cleansed Atemp)
# 
# Pass 1: 0.529  0.315  0.321  Pass 2: 0.523  0.304  0.315
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C -MONTH (Static Split)
# 
# Pass 1: 0.539  0.314  0.322  Pass 2: 0.530  0.306  0.317
# 
# ####2-MA +DoW +WK_H_AVG**3ma +N_WKD_C (Static Split)
# 
# Pass 1: 0.528  0.313  0.318  Pass 2: 0.519  0.306  0.316
# 
# ####2-MA +DoW +WK_H_AVG**5ma (Static Split)
# 
# Pass 1: 0.526  0.311  0.317  Pass 2: 0.520  0.305  0.314 
# 
# ####2-MA +DoW +WK_H_AVG** (Static Split, avg only before-3MA)
# 
# Pass 1: 0.526  0.312  0.319  Pass 2: 0.520  0.304  0.314
# 
# ####2-MA +DoW +WK_H_AVG -DAY (Static Split)
# 
# Pass 1: 0.529  0.310  0.319  Pass 2: 0.530  0.303  0.312
# Pass 1: 0.532  0.312  0.320  Pass 2: 0.529  0.302  0.311
# 
# ####2-MA +DoW +WK_H_AVG (Static Split)
# 
# Pass 1: 0.529  0.305  0.312  Pass 2: 0.531  0.299  0.308
# Pass 1: 0.531  0.305  0.313  Pass 2: 0.530  0.300  0.308
# 
# ####2-MA +HIGH_WS(35) +WK_H_AVG +N_WKD_C +WET (Static Split, Day of Week, High windspeed > 35, WD_H_AVG, Non-Workingday-Count, Wet-grounds)
# 
# Pass 1: 0.530  0.305  0.312  Pass 2: 0.530  0.301  0.309
# 
# ####2-MA +DoW +H_WS(35) +WK_H_AVG (Static Split)
# 
# Pass 1:               0.312  Pass 2:               0.307
# Pass 1: 0.530  0.305  0.312  Pass 2: 0.531  0.299  0.307
# 
# ####2-MA +DoW +HIGH_WS(35) +WK_H_AVG -WS-DAY (Static Split) --SUBMISSION 2
# 
# Pass 1: 0.321  Pass 2: 0.315  
# Pass 1: 0.321  Pass 2: 0.316  
# Pass 1: 0.322  Pass 2: 0.315  
# Pass 1: 0.323  Pass 2: 0.316
# 
# ####2-MA (Static Split, Day of Week, Remove Month)  --SUBMISSION 1
# 
# Pass 1: 0.335  Pass 2: 0.334  
# Pass 1: 0.336  Pass 2: 0.335
# 
# ####2-MA (Static Split, Day of Week)
# 
# Pass 1: 0.316  Pass 2: 0.324  
# Pass 1: 0.320  Pass 2: 0.322  
# Pass 1: 0.320  Pass 2: 0.321

# ## Submission 1
# ### RFR, Day of Week, 2-hour MA, No Month

# In[173]:
def run_submission1():
    # Execute on test

    # Prepare baseline data
    train_data, test_data, feature_names = separate_datetime(train_csv_data,
                                                             test_csv_data,
                                                             csv_feature_names)

    print 'Test data shape: ', test_data.shape

    # Prepare data with day of week feature
    train_data, test_data, feature_names = add_dayofweek(train_data,
                                                         test_data,
                                                         feature_names)


    # Remove datetime,month column for the following analyses
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME,MONTH],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    # Execute RFR
    print_div_line('RFR Initial, with Day of Week')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print


    # Reinit
    # Prepare baseline data
    train_data, test_data, feature_names = separate_datetime(train_csv_data,
                                                             test_csv_data,
                                                             csv_feature_names)

    # Prepare data with day of week feature
    train_data, test_data, feature_names = add_dayofweek(train_data,
                                                         test_data,
                                                         feature_names)

    # Prepare data with moving average features
    test_labels = np.exp(test_log_labels) - 1
    train_data, test_data, feature_names = add_ma_x(train_data, test_data, feature_names,
                                                    label_data, test_labels, 2)
    # Remove datetime,month column for the following analyses
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME,MONTH],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    print 'Test data shape: ', test_data.shape

    # Execute RFR with predicted test_labels in the test_data set
    print_div_line('RFR Pass 1, with MA')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print

    # Prepare submission
    rfr_submit(test_log_labels)


# ###Results 1.a
# File: submission - 2014-12-06 22:40:05729725.csv
# 
# Yikes! 1.96284 - I'll stick to moral support
# 
# Wait - I accidently predicted against train data - D'oh!
# 
# ###Resuls 1.b
# File: submission - 2014-12-06 23:20:03.252746.csv
# 
# Whew! Worked better this time...
# ###New Best Entry!!!
# 66 (up 674) YhatPack 0.40013 19 Sun, 07 Dec 2014 06:23:19
# 
# ###Previous Best Entry
# 142 (up 596) YhatPack 0.41743 17 Fri, 05 Dec 2014 08:26:19 (-3d)

# ## Submission 2
# ### RFR, Day of Week, High-Windspeed(35), Weekday-Hour-Avg, 2-hour MA, No Windspeed,Day

# In[171]:
def run_submission2():
    # Execute on test

    # Prepare baseline data
    train_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

    # Prepare data with day of week feature
    train_data, test_data, feature_names = add_dayofweek(train_data, test_data, feature_names)

    # Prepare data with high windspeed feature
    train_data, test_data, feature_names = add_high_windspeed(train_data, test_data, feature_names, 35)

    # Prepare data with weekday-hour avg feature
    train_data, test_data, feature_names = add_weekday_hour_avg(train_data, test_data, feature_names, label_data)

    # Remove datetime,month column for the following analyses
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME,WINDSPEED,DAY],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    # Execute RFR
    print_div_line('RFR Pass 1')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print


    # Reinit
    # Prepare baseline data
    train_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

    # Prepare data with day of week feature
    train_data, test_data, feature_names = add_dayofweek(train_data, test_data, feature_names)

    # Prepare data with high windspeed feature
    train_data, test_data, feature_names = add_high_windspeed(train_data, test_data, feature_names, 35)

    # Prepare data with weekday-hour avg feature
    train_data, test_data, feature_names = add_weekday_hour_avg(train_data, test_data, feature_names, label_data)

    # Prepare data with moving average features
    test_labels = np.exp(test_log_labels) - 1
    train_data, test_data, feature_names = add_ma_x(train_data, test_data, feature_names,
                                                    label_data, test_labels, 2)

    # Remove datetime,month column for the following analyses
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME,WINDSPEED,DAY],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    # Execute RFR with predicted test_labels in the test_data set
    print_div_line('RFR Pass 2')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print

    # Prepare submission
    rfr_submit(test_log_labels)


# ###Results 2
# File: submission - 2014-12-13 20:51:41.178842.csv
# 
# Results: 0.42474
# 
# ###Previous Best Entry
# 68 (up 73) YhatPack 0.40013 19 Sun, 07 Dec 2014 06:23:19

# ## Submission 3
# ### RFR, Day of Week, Weekday-Hour-Avg**3ma, 2-hour MA

# In[346]:
def run_submission3():
    # Execute on test

    # Add DATE PARTS
    train_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

    # ADD DAY OF WEEK
    train_data, test_data, feature_names = add_dayofweek(train_data, test_data, feature_names)

    # Add WEEKDAY-HOUR-AVERAGE with 3-month MA
    train_data, test_data, feature_names = add_weekday_hour_avg(train_data, test_data, feature_names, label_data, 3)

    # Remove DATETIME
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    # Execute RFR
    print_div_line('RFR Pass 1')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print


    # Reinit
    # Add DATE PARTS
    train_data, test_data, feature_names = separate_datetime(train_csv_data, test_csv_data, csv_feature_names)

    # Add DAY OF WEEK
    train_data, test_data, feature_names = add_dayofweek(train_data, test_data, feature_names)

    # Add WEEKDAY-HOUR-AVERAGE with 3-month MA
    train_data, test_data, feature_names = add_weekday_hour_avg(train_data, test_data, feature_names, label_data, 3)

    # Add 2-HOUR MA
    test_labels = np.exp(test_log_labels) - 1
    train_data, test_data, feature_names = add_ma_x(train_data, test_data, feature_names,
                                                    label_data, test_labels, 2)

    # Remove DATETIME
    feature_names, train_data, zero_data, test_data = remove_features([DATETIME],
                                                                     feature_names,
                                                                     train_data,
                                                                     np.zeros(train_data.shape),
                                                                     test_data)

    # Execute RFR with predicted test_labels in the test_data set
    print_div_line('RFR Pass 2')
    test_log_labels = rfr_test_predict(train_data, log_label_data, test_data, 300)
    print

    # Prepare submission
    rfr_submit(test_log_labels)


# ###Results 2
# File: submission - 2014-12-14 04:20:56.666505.csv
# 
# Results: 0.44631
# 
# ###Previous Best Entry
# 68 (up 73) YhatPack 0.40013 19 Sun, 07 Dec 2014 06:23:19

# In[347]:




# In[ ]:



