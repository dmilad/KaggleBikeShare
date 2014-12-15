
# coding: utf-8

# In[ ]:
def visualize_preds():
  ### the predictions, visually

  # In[ ]:

  plt.figure(figsize=(16,6))

  x_train = np.array([row[0] for row in data_t])[10766:]
  y_train_c = np.array([row[0] for row in train_labels_t])[10766:]
  y_train_r = np.array([row[1] for row in train_labels_t])[10766:]

  x_test = np.array([row[0] for row in data_tt])[6207:]
  y_test_1 = linear_regression_pred_labels1[6207:]
  y_test_2 = rfr_pred_labels_t[6207:]
  y_test_3 = rfra_pred_labels_t[6207:]



  plt.plot(x_train,y_train_c,color='grey',linewidth=2,label='casual')
  plt.plot(x_train,y_train_r,color='cyan',linewidth=2,label='registered')
  plt.plot(x_train, np.sum([y_train_c, y_train_r], axis=0), color = 'orange', linewidth=2, label='total count')

  plt.plot(x_test,y_test_1,color='green',alpha=0.5,linewidth=2,label='linearRegression')
  plt.plot(x_test,y_test_2,color='blue',alpha=0.5,linewidth=2,label='Random Forest')
  plt.plot(x_test,y_test_3,color='red',alpha=0.5,linewidth=2,label='Random Forest with ARIMA')


  plt.legend()
  remove_border()
  plt.show()


# In[ ]:
def visualize_diffs():
  #visualize the differences between the predicted values by different models

  #ridge_linear = ridge_pred_labels_t - linear_regression_pred_labels1
  rfr_linear = rfr_pred_labels_t - linear_regression_pred_labels1
  #rfr_ridge = rfr_pred_labels_t - ridge_pred_labels_t
  ## Should it be rfr - rfra?
  rfr_rfra = rfr_pred_labels_t - rfra_pred_labels_t
  rfra_linear = rfra_pred_labels_t - linear_regression_pred_labels1

  x = np.array([row[0] for row in data_tt])

  plt.figure(figsize=(20,10))

  plt.plot(x, rfr_rfra, color = 'black', label = 'rfr - rfrarima', alpha = 0.8)
  plt.plot(x, rfr_linear, color = 'green', label = 'rfr - linear', alpha = 0.5)
  #plt.plot(x, rfra_linear, color = 'blue', label = 'rfra - linear', alpha = 0.5)

  plt.legend()
  remove_border()
  plt.show()

def work_non_work_day_models():

  train_data_wk, train_data_nwk, train_labels_wk, train_labels_nwk = split_workday(train_data_wd_nd ,train_labels_t1, 0 ,0, 0, 0)
  dev_data_wk, dev_data_nwk, dev_labels_wk, dev_labels_nwk = split_workday(dev_data_wd_nd ,dev_labels_t1, 0 ,0, 0, 0)
  test_data_wk, test_data_nwk, test_labels_wk, test_labels_nwk = split_workday(test_data_wd_nd ,np.zeros_like(test_data_wd_nd), 0 ,0, 0, 0)


  print train_data_wk.shape, train_data_nwk.shape, train_labels_wk.shape, train_labels_nwk.shape
  column_names_wk = column_names2_nd[:5] + column_names2_nd[6:]


  # In[ ]:

  n_est = [1, 5, 10, 20, 40, 60, 100, 300]
  print "workingday"
  print "n\t\tc\tr"

  for n in n_est:
    rfrwk = RandomForestRegressor(n_estimators=n)
    rfrwk.fit(train_data_wk, train_labels_wk[:,0:2])

    rfrwk_log_pred_labels = rfrwk.predict(dev_data_wk)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(rfrwk_log_pred_labels[:,0], dev_labels_wk[:,0]), RMSLE(rfrwk_log_pred_labels[:,1], dev_labels_wk[:,1]))
    if n == 300:
      print "\nfeature importances:"
      for name, feature in zip(column_names_wk, rfrwk.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)

  print "\nnon-workingday"
  print "n\t\tc\tr"

  for n in n_est:
    rfrnwk = RandomForestRegressor(n_estimators=n)
    rfrnwk.fit(train_data_nwk, train_labels_nwk[:,0:2])

    rfrnwk_log_pred_labels = rfrnwk.predict(dev_data_nwk)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(rfrnwk_log_pred_labels[:,0], dev_labels_nwk[:,0]), RMSLE(rfrnwk_log_pred_labels[:,1], dev_labels_nwk[:,1]))
    if n == 300:
      print "\nfeature importances:"
      for name, feature in zip(column_names_wk, rfrnwk.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)


  # In[ ]:

  rfrwk = RandomForestRegressor(n_estimators=300)

  rfrwk.fit(np.vstack((train_data_wk, dev_data_wk)), np.vstack((train_labels_wk[:,0:2],dev_labels_wk[:,0:2])))
  rfr_log_pred_labels_wk = rfrwk.predict(test_data_wk)

  rfr_casual_pred_wk = np.exp(rfr_log_pred_labels_wk[:,0]) - 1
  rfr_registered_pred_wk = np.exp(rfr_log_pred_labels_wk[:,1]) - 1

  rfr_pred_labels_wk = rfr_casual_pred_wk + rfr_registered_pred_wk


  rfrnwk = RandomForestRegressor(n_estimators=300)

  rfrnwk.fit(np.vstack((train_data_nwk, dev_data_nwk)), np.vstack((train_labels_nwk[:,0:2],dev_labels_nwk[:,0:2])))
  rfr_log_pred_labels_nwk = rfrnwk.predict(test_data_nwk)

  rfr_casual_pred_nwk = np.exp(rfr_log_pred_labels_nwk[:,0]) - 1
  rfr_registered_pred_nwk = np.exp(rfr_log_pred_labels_nwk[:,1]) - 1

  rfr_pred_labels_nwk = rfr_casual_pred_nwk + rfr_registered_pred_nwk

  print rfr_pred_labels_wk.shape

  for i in range(len(test_data_wd_nd)):
    if test_data_wd_nd[i][5] == 1:
      try:
        wk_nwk = np.append(wk_nwk, rfr_pred_labels_wk[0])
      except:
        wk_nwk = rfr_pred_labels_wk[0]
      rfr_pred_labels_wk = np.delete(rfr_pred_labels_wk, 0)
    else:
      try:
        wk_nwk = np.append(wk_nwk, rfr_pred_labels_nwk[0])
      except:
        wk_nwk = rfr_pred_labels_nwk[0]
      rfr_pred_labels_nwk = np.delete(rfr_pred_labels_nwk, 0)    
      
  print wk_nwk.shape

  #write_submission(wk_nwk)

def visualize_models():
  ### the various models visualized

  # In[ ]:

  """plt.figure(figsize=(20,6))

  x_train = np.array([row[0] for row in data_t])
  y_train_c = np.array([row[0] for row in train_labels_t])
  y_train_r = np.array([row[1] for row in train_labels_t])

  x_test = np.array([row[0] for row in data_tt])
  y_test = pred_labels1 


  #plt.plot(x_train,y_train_c,color='grey',label='casual')
  #plt.plot(x_train,y_train_r,color='cyan', alpha=0.5, label='registered')
  plt.plot(x_train, np.sum([y_train_c, y_train_r], axis=0), color = 'orange', linewidth=2, label='total count')

  plt.plot(x_test,y_test, color='red',label='predicted')

  plt.legend()
  remove_border()
  plt.show()

  """

  plt.figure(figsize=(16,6))

  x_train = np.array([row[0] for row in data_t])[10766:]
  y_train_c = np.array([row[0] for row in train_labels_t])[10766:]
  y_train_r = np.array([row[1] for row in train_labels_t])[10766:]

  x_test = np.array([row[0] for row in data_tt])[6207:]
  y_test_1 = linear_regression_pred_labels1[6207:]
  y_test_2 = ridge_pred_labels_t[6207:]
  y_test_3 = rfr_pred_labels_t[6207:]



  plt.plot(x_train,y_train_c,color='grey',linewidth=2,label='casual')
  plt.plot(x_train,y_train_r,color='cyan',linewidth=2,label='registered')
  plt.plot(x_train, np.sum([y_train_c, y_train_r], axis=0), color = 'orange', linewidth=2, label='total count')

  plt.plot(x_test,y_test_1,color='red',linewidth=2,label='linearRegression')
  plt.plot(x_test,y_test_2,color='blue',alpha=0.5,linewidth=2,label='Ridge')
  plt.plot(x_test,y_test_3,color='green',alpha=0.5,linewidth=2,label='Random Forest')


  plt.legend()
  remove_border()
  plt.show()

def visualize_differences():
  ### what does the difference between the predicted values by each model look like?

  # In[ ]:

  #visualize the differences between the predicted values by different models

  ridge_linear = ridge_pred_labels_t - linear_regression_pred_labels1
  rfr_linear = rfr_pred_labels_t - linear_regression_pred_labels1
  rfr_ridge = rfr_pred_labels_t - ridge_pred_labels_t

  x = np.array([row[0] for row in data_tt])

  plt.figure(figsize=(20,10))

  plt.plot(x, ridge_linear, color = 'black', label = 'ridge - linear')
  plt.plot(x, rfr_linear, color = 'green', label = 'rfr - linear', alpha = 0.5)
  #plt.plot(x, rfr_ridge, color = 'blue', label = 'rfr - ridge', alpha = 0.5)

  plt.legend()
  remove_border()
  plt.show()

def plot_outputs():
  ### how do the output values look over time?

  # In[ ]:

  plt.figure(figsize=(24,6))

  x_train = np.array([row[0] for row in data_t])
  y_train_c = np.array([row[0] for row in train_labels_t])
  y_train_r = np.array([row[1] for row in train_labels_t])

  x_test = np.array([row[0] for row in data_tt])
  y_test = np.random.randint(1000, size=x_test.shape) 
  #i am creating a random y value here, just to show the overlap between train and test

  plt.plot(x_train,y_train_c,color='grey',label='casual')
  plt.plot(x_train,y_train_r,color='cyan', alpha=0.5, label='registered')

  plt.plot(x_test,y_test, color='red',alpha=0.1,label='test (random)')

  plt.legend()
  remove_border()
  plt.show()

def outputs_zoom():
  ### let's zoom in to the rightmost part of the chart above, looking only at december 2012

  # In[ ]:

  plt.figure(figsize=(24,6))

  x_train = np.array([row[0] for row in data_t])[10430:]
  y_train_c = np.array([row[0] for row in train_labels_t])[10430:]
  y_train_r = np.array([row[1] for row in train_labels_t])[10430:]


  plt.plot(x_train,y_train_c,color='grey',linewidth=2,label='casual')
  plt.plot(x_train,y_train_r,color='cyan',linewidth=2,label='registered')
  plt.plot(x_train, np.sum([y_train_c, y_train_r], axis=0), color = 'orange', linewidth=2, label='total count')

  plt.legend()
  remove_border()
  plt.show()

def output_correlations_plots():
  ### is there any relationship between 'casual' and 'registered'? any outliers? 

  # In[ ]:

  fig1 = plt.figure(figsize=(10,5))

  plt.scatter(np.append(train_labels[:,1],dev_labels[:,1]),np.append(train_labels[:,0],dev_labels[:,0]),marker='.')
  plt.xlabel('registered')
  plt.ylabel('casual')
  plt.axis('equal')
  plt.xlim(0,1000)
  plt.ylim(0,500)
  remove_border()
  plt.show()

  print "correlation matrix between casual and registered:" 
  print np.corrcoef(np.vstack((train_labels[:,0:2],dev_labels[:,0:2])), rowvar=0)
  print "\n"

  numBins = 50
  plt.figure(figsize=(24,4))
  plt.subplot(131)
  remove_border()
  plt.hist(np.append(train_labels[:,0],dev_labels[:,0]),numBins,color='grey',alpha=0.8,label='casual')
  plt.legend()
  plt.subplot(132)
  remove_border()
  plt.hist(np.append(train_labels[:,1],dev_labels[:,1]),numBins,color='cyan',alpha=0.5,label='registered')
  plt.legend()
  plt.subplot(133)
  remove_border()
  plt.hist(np.append(train_labels[:,2],dev_labels[:,2]),numBins,color='orange',alpha=0.8,label='total count')
  plt.legend()
  plt.show()


  # there seems to be two different clusters of relationship between casual and registered. is this important? can we figure out what is different between these two groups? would k-means help here?
  # 
  # there doesn't appear to be any outliers in the train+dev labels, so we are good there.


def ridge_regression_run():
  # #Ridge regression
  # ##separate models for casual and registered

  # In[ ]:

  alpha = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]

  print "a\t\tc\tr"

  for a in alpha:
    R = Ridge(alpha = a)

    R.fit(train_data_t3n, train_labels_t1[:,0])
    log_pred_labels_c = R.predict(dev_data_t3n)

    R.fit(train_data_t3n, train_labels_t1[:,1])
    log_pred_labels_r = R.predict(dev_data_t3n)

    print "%8.3f\t%7.5f\t%7.5f" % (a,RMSLE(log_pred_labels_c, dev_labels_t1[:,0]), RMSLE(log_pred_labels_r, dev_labels_t1[:,1]))



  # In[ ]:

  R = Ridge(alpha = 10)

  R.fit(np.vstack((train_data_r3n,dev_data_r3n)), np.append(train_labels_t1[:,0],dev_labels_t1[:,0]))
  log_pred_labels_c = R.predict(test_data_r3n)

  R = Ridge(alpha = 0.01)

  R.fit(np.vstack((train_data_r3n,dev_data_r3n)), np.append(train_labels_t1[:,1],dev_labels_t1[:,1]))
  log_pred_labels_r = R.predict(test_data_r3n)

  casual_pred = np.exp(log_pred_labels_c) - 1
  registered_pred = np.exp(log_pred_labels_r) - 1

  print "%d predicted casual values <= 0" % len(casual_pred[casual_pred<0])

  pred_labels = casual_pred + registered_pred

  print "%d predicted count values <= 0" % len(pred_labels[pred_labels<0])

  ridge_pred_labels_t = pred_labels
  for i in range(len(pred_labels)):
    if pred_labels[i] < 0:
      ridge_pred_labels_t[i] = 0

  print ridge_pred_labels_t.shape

  #write_submission(pred_labels_t) only submitted this one

  casual_pred_t = casual_pred
  for i in range(len(casual_pred)):
    if casual_pred[i] < 0:
      casual_pred_t[i] = 0

  pred_labels2 = casual_pred_t + registered_pred

  #write_submission(pred_labels2)

def lasso_regression_run():
  # #Lasso
  # ##separate models for casual and registered

  # In[ ]:

  alpha = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]

  print "a\t\tc\tr"

  for a in alpha:
    L = Lasso(alpha = a)

    L.fit(train_data_r3n, train_labels_t1[:,0])
    log_pred_labels_c = L.predict(dev_data_r3n)

    L.fit(train_data_t3n, train_labels_t1[:,1])
    log_pred_labels_r = L.predict(dev_data_r3n)

    print "%8.3f\t%7.5f\t%7.5f" % (a,RMSLE(log_pred_labels_c, dev_labels_t1[:,0]), RMSLE(log_pred_labels_r, dev_labels_t1[:,1]))


  # the best Lasso model for each output is worse than Ridge, so we won't pursue Lasso further.

def split_workday(data, labels, wk_data, nwk_data, wk_labels, nwk_labels):
  wk_data = np.array([])
  nwk_data = np.array([])
  wk_labels = np.array([])
  nwk_labels = np.array([])
  
  for i in range(len(data)):
    if data[i][5] == 1:
      try:
        wk_data = np.vstack((wk_data, np.append(data[i][:5],data[i][6:])))
        wk_labels = np.vstack((wk_labels, np.append(labels[i][:5],labels[i][6:])))
      except:
        wk_data = np.append(data[i][:5],data[i][6:])
        wk_labels = np.append(labels[i][:5],labels[i][6:])
    else:
      try:
        nwk_data = np.vstack((nwk_data, np.append(data[i][:5],data[i][6:])))
        nwk_labels = np.vstack((nwk_labels, np.append(labels[i][:5],labels[i][6:])))
      except:
        nwk_data = np.append(data[i][:5],data[i][6:])
        nwk_labels = np.append(labels[i][:5],labels[i][6:])
        
  return wk_data, nwk_data, wk_labels, nwk_labels


def work_non_work_day_models():

  train_data_wk, train_data_nwk, train_labels_wk, train_labels_nwk = split_workday(train_data_wd_nd ,train_labels_t1, 0 ,0, 0, 0)
  dev_data_wk, dev_data_nwk, dev_labels_wk, dev_labels_nwk = split_workday(dev_data_wd_nd ,dev_labels_t1, 0 ,0, 0, 0)
  test_data_wk, test_data_nwk, test_labels_wk, test_labels_nwk = split_workday(test_data_wd_nd ,np.zeros_like(test_data_wd_nd), 0 ,0, 0, 0)


  print train_data_wk.shape, train_data_nwk.shape, train_labels_wk.shape, train_labels_nwk.shape
  column_names_wk = column_names2_nd[:5] + column_names2_nd[6:]


  # In[ ]:

  n_est = [1, 5, 10, 20, 40, 60, 100, 300]
  print "workingday"
  print "n\t\tc\tr"

  for n in n_est:
    rfrwk = RandomForestRegressor(n_estimators=n)
    rfrwk.fit(train_data_wk, train_labels_wk[:,0:2])

    rfrwk_log_pred_labels = rfrwk.predict(dev_data_wk)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(rfrwk_log_pred_labels[:,0], dev_labels_wk[:,0]), RMSLE(rfrwk_log_pred_labels[:,1], dev_labels_wk[:,1]))
    if n == 300:
      print "\nfeature importances:"
      for name, feature in zip(column_names_wk, rfrwk.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)

  print "\nnon-workingday"
  print "n\t\tc\tr"

  for n in n_est:
    rfrnwk = RandomForestRegressor(n_estimators=n)
    rfrnwk.fit(train_data_nwk, train_labels_nwk[:,0:2])

    rfrnwk_log_pred_labels = rfrnwk.predict(dev_data_nwk)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(rfrnwk_log_pred_labels[:,0], dev_labels_nwk[:,0]), RMSLE(rfrnwk_log_pred_labels[:,1], dev_labels_nwk[:,1]))
    if n == 300:
      print "\nfeature importances:"
      for name, feature in zip(column_names_wk, rfrnwk.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)


  # In[ ]:

  rfrwk = RandomForestRegressor(n_estimators=300)

  rfrwk.fit(np.vstack((train_data_wk, dev_data_wk)), np.vstack((train_labels_wk[:,0:2],dev_labels_wk[:,0:2])))
  rfr_log_pred_labels_wk = rfrwk.predict(test_data_wk)

  rfr_casual_pred_wk = np.exp(rfr_log_pred_labels_wk[:,0]) - 1
  rfr_registered_pred_wk = np.exp(rfr_log_pred_labels_wk[:,1]) - 1

  rfr_pred_labels_wk = rfr_casual_pred_wk + rfr_registered_pred_wk


  rfrnwk = RandomForestRegressor(n_estimators=300)

  rfrnwk.fit(np.vstack((train_data_nwk, dev_data_nwk)), np.vstack((train_labels_nwk[:,0:2],dev_labels_nwk[:,0:2])))
  rfr_log_pred_labels_nwk = rfrnwk.predict(test_data_nwk)

  rfr_casual_pred_nwk = np.exp(rfr_log_pred_labels_nwk[:,0]) - 1
  rfr_registered_pred_nwk = np.exp(rfr_log_pred_labels_nwk[:,1]) - 1

  rfr_pred_labels_nwk = rfr_casual_pred_nwk + rfr_registered_pred_nwk

  print rfr_pred_labels_wk.shape

  for i in range(len(test_data_wd_nd)):
    if test_data_wd_nd[i][5] == 1:
      try:
        wk_nwk = np.append(wk_nwk, rfr_pred_labels_wk[0])
      except:
        wk_nwk = rfr_pred_labels_wk[0]
      rfr_pred_labels_wk = np.delete(rfr_pred_labels_wk, 0)
    else:
      try:
        wk_nwk = np.append(wk_nwk, rfr_pred_labels_nwk[0])
      except:
        wk_nwk = rfr_pred_labels_nwk[0]
      rfr_pred_labels_nwk = np.delete(rfr_pred_labels_nwk, 0)    
      
  print wk_nwk.shape

  #write_submission(wk_nwk)


### the improvement is not significant. 7 spot improvement.

def atemp_outlier_fix(dev_data):
  """
  fix outliers for atemp
  they all fall on august 17, 2012
  """
  dev_data_new = dev_data
  for i in range(len(dev_data)):
    if dev_data[i][8]>20 and dev_data[i][9]==12.12:
      dev_data_new[i][9] = dev_data[i][8] + 3
  
  return dev_data_new



def build_day_dict(day_dict, feature, data, cond = 0):
  """
  build dictionary that counts number of consecutive days with same value of feature. 
  for example, if two consecutive days both have values workingday = 1, day_dict would have a 2 for each of those days
  you should feed combined (train,dev,test) to data
  for non-working-day-count (3-day-weekend) feature, set feature to workingday
  """
  d_dict = {}
  day_arr, feature_arr = [], []
  for n in range(len(data)):
    if datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2])) not in d_dict:
      d_dict[datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]))] = data[n][feature]

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


def build_day_feature(day_dict, train_data, dev_data, test_data):
  """
  append new feature based on day_dict to data
  """
  train_data_r = np.zeros((train_data.shape[0], train_data.shape[1]+1))
  dev_data_r = np.zeros((dev_data.shape[0], dev_data.shape[1]+1))
  test_data_r = np.zeros((test_data.shape[0], test_data.shape[1]+1))
  
  for i in range(len(train_data)):
    train_data_r[i] = np.append(train_data[i], day_dict[str(datetime.datetime(int(train_data[i][0]), int(train_data[i][1]), int(train_data[i][2])))])

  for i in range(len(dev_data)):
    dev_data_r[i] = np.append(dev_data[i], day_dict[str(datetime.datetime(int(dev_data[i][0]), int(dev_data[i][1]), int(dev_data[i][2])))])

  for i in range(len(test_data)):
    test_data_r[i] = np.append(test_data[i], day_dict[str(datetime.datetime(int(test_data[i][0]), int(test_data[i][1]), int(test_data[i][2])))])
  
  return train_data_r, dev_data_r, test_data_r



def build_past_period_dict(p_dict, period, feature, threshold, data):
  """
  build dictionary that counts number of occurances of a certain value in the past x hours.
  for example, count number of hours with weather > 2 in the last 3 hours.
  you should feed combined (train,dev,test) to data
  for wet-grounds, set period to 3, feature to weather, threshold to 2
  """
  return_dict = {}
  
  for n in range(len(data)):
    if datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])) not in p_dict:
      p_dict[datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3]))] = data[n][feature]

  ordered_dict = collections.OrderedDict(sorted(p_dict.items()))
  
  for key, value in ordered_dict.iteritems():
    date_list = [key - datetime.timedelta(hours = h_i+1) for h_i in range(period)]
    
    count = 0
    for d in date_list:
      if d in ordered_dict and ordered_dict[d] > threshold:
        count += 1
    
    return_dict[key] = count
        
  return return_dict


def build_period_feature(period_dict, train_data, dev_data, test_data):
  """
  append new feature based on period_dict to data
  """
  train_data_r = np.zeros((train_data.shape[0], train_data.shape[1]+1))
  dev_data_r = np.zeros((dev_data.shape[0], dev_data.shape[1]+1))
  test_data_r = np.zeros((test_data.shape[0], test_data.shape[1]+1))
  
  for i in range(len(train_data)):
    train_data_r[i] = np.append(train_data[i], period_dict[datetime.datetime(int(train_data[i][0]), int(train_data[i][1]), int(train_data[i][2]), int(train_data[i][3]))])

  for i in range(len(dev_data)):
    dev_data_r[i] = np.append(dev_data[i], period_dict[datetime.datetime(int(dev_data[i][0]), int(dev_data[i][1]), int(dev_data[i][2]), int(dev_data[i][3]))])

  for i in range(len(test_data)):
    test_data_r[i] = np.append(test_data[i], period_dict[datetime.datetime(int(test_data[i][0]), int(test_data[i][1]), int(test_data[i][2]), int(test_data[i][3]))])
  
  return train_data_r, dev_data_r, test_data_r

# In[ ]:

get_ipython().magic(u'matplotlib inline')

import os
import csv
import collections

import datetime
import math

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
  #now=str(datetime.datetime.now()) #windows can't have colons in filename
  now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
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


#log-transform 'casual' and 'registered'

def log_trans_labels(labels):
  """
  transform labels
  
  """
  return_labels = np.array([])
  for row in labels:
    logc = np.log(row[0]+1)
    logr = np.log(row[1]+1)
    logt = np.log(row[2]+1)
    
    return_row = np.array([logc, logr, logt])
    try:
      return_labels = np.vstack((return_labels, return_row))
    except:
      return_labels = return_row
  return return_labels


## Load Data

# In[ ]:

#prepare train and test data

#read in train data
with open('train.csv','rb') as rfile:
  reader = csv.reader(rfile)
  data = [i for i in reader] 

#get rid of column headers
column_names = data[0][1:-3]
column_names = ['year','month','day','hour'] + column_names
data_t = data[1:]

train_data_t = []
train_labels_t = []

#change data types
for row in data_t:
  row[0] = datetime.datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S')
  row[1] = int(row[1])
  row[2] = int(row[2])
  row[3] = int(row[3])
  row[4] = int(row[4])
  row[5] = float(row[5])
  row[6] = float(row[6])
  row[7] = int(row[7])
  row[8] = float(row[8])
  row[9] = int(row[9])
  row[10] = int(row[10])
  row[11] = int(row[11])
  #break down datetime to year, month, day, hour
  #jh: adding original datetime value at end of row
  #train_data_t.append([row[0].year,row[0].month,row[0].day,row[0].hour,row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[0]])
  train_data_t.append([row[0].year,row[0].month,row[0].day,row[0].hour,row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]])
  train_labels_t.append([row[9],row[10],row[11]])
  
train_data = np.array(train_data_t)
train_labels = np.array(train_labels_t)
print train_data.shape
print train_labels.shape

##do all of the above for test data (except test data doesnt have labels)
with open('test.csv','rb') as rfile:
  reader = csv.reader(rfile)
  data = [i for i in reader]

#get rid of column headers
data_tt = data[1:]

test_data=[]

for row in data_tt:
  row[0] = datetime.datetime.strptime(row[0],'%Y-%m-%d %H:%M:%S')
  row[1] = int(row[1])
  row[2] = int(row[2])
  row[3] = int(row[3])
  row[4] = int(row[4])
  row[5] = float(row[5])
  row[6] = float(row[6])
  row[7] = int(row[7])
  row[8] = float(row[8])
  #break down datetime to year, month, day, hour
  #jh: adding original datetime value at end of row
  #test_data.append([row[0].year,row[0].month,row[0].day,row[0].hour,row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[0]])
  test_data.append([row[0].year,row[0].month,row[0].day,row[0].hour,row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8]])

test_data = np.array(test_data)
print test_data.shape

## Split Approaches

# ###split train and dev by day of month (first 16 days train, following 3 days dev)
# training set contains the first 19 days of data for each month, and the test contains the rest. Given the time series nature of the data set, it makes sense to split the training set so that the holdout set is after the training set. (16 days train, 3 days dev, rest test)

# In[ ]:

train, tlabel = [], []
dev, dlabel = [], []
for i in range(len(train_data)):
  if train_data[i][2] < 17:
    train.append(train_data[i])
    tlabel.append(train_labels[i])
  else:
    dev.append(train_data[i])
    dlabel.append(train_labels[i])

train_data = np.array(train)
train_labels = np.array(tlabel)
dev_data = np.array(dev)
dev_labels = np.array(dlabel)


print train_data.shape, train_labels.shape
print dev_data.shape, dev_labels.shape


train_labels_t1 = log_trans_labels(train_labels)
print train_labels_t1.shape
dev_labels_t1 = log_trans_labels(dev_labels)
print dev_labels_t1.shape

data = np.vstack((train_data, dev_data))
labels = np.vstack((train_labels, dev_labels))
colors = ['grey', 'cyan']

## Visualizing the data

# ###what do the different attributes look like?
# 
# even though we are going to use on train_data to train the model, for submission we will be using both train and dev.
# 

# In[ ]:
def print_hists():
  hists = plt.figure(figsize=(24,12))

  numBins = 60
  for i in range(len(train_data[0])):
    plt.subplot(3,4,i+1)
    remove_border()
    plt.hist(np.append(train_data[:,i],dev_data[:,i]),numBins,color='green',alpha=0.8)
    plt.hist(test_data[:,i],numBins,color='red',alpha=0.6)
    plt.title(column_names[i])
  plt.legend(['train+dev','test'],loc=1)
  plt.show()


### The bar charts below depict the average of casual, registered, and total count for each categorical variable

# In[ ]:
def print_averages_of_all():
  colors = ['grey', 'cyan']

  bars = plt.figure(figsize=(24,12))

  data = np.vstack((train_data, dev_data))
  labels = np.vstack((train_labels, dev_labels))

  cat_vars = range(12)

  for c in cat_vars:
    plt.subplot(3,4,c+1)
    remove_border()
    plt.title(column_names[c])
        
    cat_set = []
    for i in np.vstack((data, test_data))[:,c]:
      if int(i) not in cat_set:
        cat_set.append(int(i))

    bar_h0 = []
    bar_h1 = []
    bar_hmissing = []
    for i in cat_set:
      s0 = 0
      s1 = 0
      cnt = 0
      for di in range(len(data[:,c])):
        if int(data[di,c]) == i:
          cnt += 1
          s0 += labels[di,0]
          s1 += labels[di,1]
      if cnt > 0: 
        bar_h0.append(s0/cnt)
        bar_h1.append(s1/cnt)
        bar_hmissing.append(0)
      else:
        bar_h0.append(0)
        bar_h1.append(0)
        bar_hmissing.append(10)

    plt.bar(cat_set, bar_h0, color = colors[0])
    plt.bar(cat_set, bar_h1, bottom = bar_h0, color = colors[1])
    plt.bar(cat_set, bar_hmissing, color = 'red')
    
  plt.legend(['casual','registered','only in test'],loc=1)        
  plt.show()



# #there are some outliers here
# ##they all belong to the same day. in fact the entire 24 hours of august 17, 2012 have bad atemp data
# ###create a function to change atemp to temp value

# In[ ]:
def print_outliers_atemp():
  plt.figure(figsize = (10,10))
  plt.scatter(np.vstack((train_data, dev_data, test_data))[:,8], np.vstack((train_data, dev_data, test_data))[:,9], alpha = 0.2)
  plt.xlabel('temp')
  plt.ylabel('atemp')
  plt.show()

def print_outliers_atemp_dist():
  data_orig = dev_data
  labels_orig = dev_labels

  print data_orig.shape

  data_out = np.zeros((0,12))
  labels_out = np.zeros((0,3))

  for i in range(len(data_orig)):
    if data_orig[i][8]>20 and data_orig[i][9]==12.12:
      data_out = np.vstack((data_out, data_orig[i]))
      labels_out = np.vstack((labels_out, labels_orig[i]))

  print data_out.shape

  plt.figure(figsize = (20,10))
  cat_vars = range(12)

  for c in cat_vars:
    plt.subplot(3,4,c+1)
    remove_border()
    plt.title(column_names[c])
        
    cat_set = []
    for i in data[:,c]:
      if int(i) not in cat_set:
        cat_set.append(int(i))

    bar_h0 = []
    bar_h1 = []
    bar_hmissing = []
    for i in cat_set:
      s0 = 0
      s1 = 0
      cnt = 0
      for di in range(len(data_out[:,c])):
        if int(data_out[di,c]) == i:
          cnt += 1
          s0 += labels_out[di,0]
          s1 += labels_out[di,1]
      if cnt > 0: 
        bar_h0.append(s0/cnt)
        bar_h1.append(s1/cnt)
        bar_hmissing.append(0)
      else:
        bar_h0.append(0)
        bar_h1.append(0)
        bar_hmissing.append(10)

    plt.bar(cat_set, bar_h0, color = colors[0])
    plt.bar(cat_set, bar_h1, bottom = bar_h0, color = colors[1])
    plt.bar(cat_set, bar_hmissing, color = 'red')
    
  plt.legend(['casual','registered','only in test'],loc=1)        
  plt.show()

def atemp_outlier_fix(dev_data):
  """
  fix outliers for atemp
  they all fall on august 17, 2012
  """
  dev_data_new = dev_data
  for i in range(len(dev_data)):
    if dev_data[i][8]>20 and dev_data[i][9]==12.12:
      dev_data_new[i][9] = dev_data[i][8] + 3
  
  return dev_data_new


# In[ ]:
def print_humid_atemp():
  plt.figure(figsize = (16,8))

  plt.subplot(121)
  plt.scatter(data[:,9], data[:,10], alpha = 0.1)
  plt.xlabel('atemp')
  plt.ylabel('humidity')
  plt.subplot(122)
  plt.scatter(data[:,8], data[:,10], alpha = 0.1)
  plt.xlabel('temp')
  plt.ylabel('humidity')

  plt.show()

def print_atemp_wind():
  plt.figure(figsize = (16,8))

  plt.subplot(121)
  plt.scatter(data[:,9], data[:,11], alpha = 0.1)
  plt.xlabel('atemp')
  plt.ylabel('windspeed')
  plt.subplot(122)
  plt.scatter(data[:,8], data[:,11], alpha = 0.1)
  plt.xlabel('temp')
  plt.ylabel('windspeed')

  plt.show()


# In[ ]:
def print_weather_time():
  plt.figure(figsize=(24,6))

  x_train = np.array([row[0] for row in data_t])

  y_holiday = np.array([row[2] for row in data_t])
  y_workingday = np.array([row[3] for row in data_t])
  y_weather = np.array([row[4] for row in data_t])

  #x_test = np.array([row[0] for row in data_tt])
  #y_test = np.random.randint(1000, size=x_test.shape) 
  #i am creating a random y value here, just to show the overlap between train and test

  #plt.scatter(x_train,y_workingday,color='red', label='workingday', alpha = 0.5, marker = '.')
  plt.scatter(x_train,y_weather,color='blue',label='weather', alpha = 1, marker = '.')

  print np.sum([np.array([row[4] for row in data_t]) == 3])

  #plt.plot(x_test,y_test, color='red',alpha=0.1,label='test (random)')

  plt.legend()
  remove_border()
  plt.show()


## data prep functions

# for regression

# In[ ]:


def dummy_var_season_and_weather(data):
  """
  create dummy variables for the weather and season attributes for train, dev, and test
  """
  return_data = np.array([])
  for row in data:
    
    season = np.array([0, 0, 0])
    if row[4] > 1:
      season[row[4]-2] = 1
    
    weather = np.array([0, 0, 0])
    if row[7] > 1:
      weather[row[7]-2] = 1
    
    return_row = np.append(row, weather)
    return_row = np.append(return_row, season)
    return_row = np.delete(return_row,[4,7])
    try:
      return_data = np.vstack((return_data,return_row))
    except:
      return_data = return_row

  return return_data


def day_of_week_as_6_dummies(data):
  """
  create 6 dummy variables for working day
  """
  return_data = np.array([])
  for row in data:
    
    dayofweek = np.zeros(6, dtype=np.int)
    w = datetime.date(int(row[0]), int(row[1]), int(row[2])).weekday()
    if  w > 0:
      dayofweek[w - 1] = 1
    
    return_row = np.append(row, dayofweek)
    
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
      
  return return_data


def dummy_var_all_date_attrs(data):
  """
  create dummy variables for the month, day, and hour attributes
  """
  return_data = np.array([])
  for row in data:
    
    year = np.zeros(1, dtype=np.int)
    if row[0] > 2013:
      year[0] = 1
    
    month = np.zeros(11, dtype=np.int)
    if row[1] > 1:
      month[row[1]-2] = 1
    
    day = np.zeros(30, dtype=np.int)
    if row[2] > 1:
      day[row[2]-2] = 1
      
    hour = np.zeros(23, dtype=np.int)
    if row[3] > 0:
      hour[row[3]-2] = 1
    
    return_row = np.append(row, year)
    return_row = np.append(return_row, month)
    return_row = np.append(return_row, day)
    return_row = np.append(return_row, hour)
    return_row = np.delete(return_row,[0,1,2,3])
    
    try:
      return_data = np.vstack((return_data,return_row))
    except:
      return_data = return_row

  return return_data


def trans_humidity_wind(data):
  """
  transform humidity and windspeed
  """
  return_data = np.array([])
  for row in data:
    return_row = row
    return_row[4], return_row[5] = row[4]**2, row[5]**0.5
    
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
    
  return return_data



def mean_norm_4_cont_features(data):
  """
  mean normalize the 4 continuous features
  """
  data_to_norm = data[:,2:6]
  normed = preprocessing.scale(data_to_norm)
  data[:,2:6] = normed
  return data



def plot_log_trans_outputs():
  #plot the transformed labels
  numBins = 50
  plt.figure(figsize=(24,4))
  plt.subplot(131)
  remove_border()
  plt.hist(np.append(train_labels_t1[:,0],dev_labels_t1[:,0]),numBins,color='grey',alpha=0.8,label='casual')
  plt.legend()
  plt.subplot(132)
  remove_border()
  plt.hist(np.append(train_labels_t1[:,1],dev_labels_t1[:,1]),numBins,color='cyan',alpha=0.8,label='registered')
  plt.legend()
  plt.subplot(133)
  remove_border()
  plt.hist(np.append(train_labels_t1[:,2],dev_labels_t1[:,2]),numBins,color='orange',alpha=0.8,label='total count')
  plt.legend()
  plt.show()


#### i think that, given the different distributions for casual and registered, it makes sense to develop separate models for each and then combine the results to find the total count.

# In[ ]:

##For Regression:
def regression_data_prep_and_plot():
  #dummy variable the sesason and weather attributes
  train_data_r1 = dummy_var_season_and_weather(train_data)
  dev_data_r1 = dummy_var_season_and_weather(dev_data)
  test_data_r1 = dummy_var_season_and_weather(test_data)

  print train_data_r1.shape
  col_sw = column_names[:4] + column_names[5:7] + column_names[8:]+['s 2-1', 's 3-1', 's 4-1', 'w 2-1', 'w 3-1', 'w 4-1']
  #print col_sw

  #add day of the week to the featureset via 6 dummy variables
  train_data_r1w = day_of_week_as_6_dummies(train_data_r1)
  dev_data_r1w = day_of_week_as_6_dummies(dev_data_r1)
  test_data_r1w = day_of_week_as_6_dummies(test_data_r1)

  print train_data_r1w.shape

  #dummy var all date attributes
  train_data_r2 = dummy_var_all_date_attrs(train_data_r1w)
  dev_data_r2 = dummy_var_all_date_attrs(dev_data_r1w)
  test_data_r2 = dummy_var_all_date_attrs(test_data_r1w)

  print train_data_r2.shape

  print "\nOriginal:"
  #transform humidity and windspeed
  figure = plt.figure(figsize=(8,3))

  numBins = 100
  names = ['humidity','windspeed']
  for i in range(4,6):
    plt.subplot(1,2,i-3)
    remove_border()
    plt.hist(np.append(train_data_r2[:,i],dev_data_r2[:,i]),numBins,color='green',alpha=0.8)
    plt.hist(test_data_r2[:,i],numBins,color='red',alpha=0.6)
    plt.title(names[i-4])
  plt.legend(['train+dev','test'],loc=1)
  plt.show()


  train_data_r3 = trans_humidity_wind(train_data_r2)
  dev_data_r3 = trans_humidity_wind(dev_data_r2)
  test_data_r3 = trans_humidity_wind(test_data_r2)

  print "Transformed:"
  figure = plt.figure(figsize=(8,3))

  names = ['humidity','windspeed']
  for i in range(4,6):
    plt.subplot(1,2,i-3)
    remove_border()
    plt.hist(np.append(train_data_r3[:,i],dev_data_r3[:,i]),numBins,color='green',alpha=0.8)
    plt.hist(test_data_r3[:,i],numBins,color='red',alpha=0.6)
    plt.title(names[i-4])
  plt.legend(['train+dev','test'],loc=1)
  plt.show()

  #mean normalize the 4 continuous variables
  train_data_r3n = mean_norm_4_cont_features(train_data_r3)
  dev_data_r3n = mean_norm_4_cont_features(dev_data_r3)
  test_data_r3n = mean_norm_4_cont_features(test_data_r3)

  print "Mean-normalized:"

  figure = plt.figure(figsize=(16,3))

  numBins = 100
  names = ['temp','atemp','humidity','windspeed']
  for i in range(2,6):
    plt.subplot(1,4,i-1)
    remove_border()
    plt.hist(np.append(train_data_r3n[:,i],dev_data_r3n[:,i]),numBins,color='green',alpha=0.8)
    plt.hist(test_data_r3n[:,i],numBins,color='red',alpha=0.6)
    plt.title(names[i-4])
  plt.legend(['train+dev','test'],loc=1)
  plt.show()


## modeling

### linear regression

# In[ ]:
def run_linear_regression():
  #dummy variable the sesason and weather attributes
  train_data_r1 = dummy_var_season_and_weather(train_data)
  dev_data_r1 = dummy_var_season_and_weather(dev_data)
  test_data_r1 = dummy_var_season_and_weather(test_data)

  col_sw = column_names[:4] + column_names[5:7] + column_names[8:]+['s 2-1', 's 3-1', 's 4-1', 'w 2-1', 'w 3-1', 'w 4-1']

  #add day of the week to the featureset via 6 dummy variables
  train_data_r1w = day_of_week_as_6_dummies(train_data_r1)
  dev_data_r1w = day_of_week_as_6_dummies(dev_data_r1)
  test_data_r1w = day_of_week_as_6_dummies(test_data_r1)

  #dummy var all date attributes
  train_data_r2 = dummy_var_all_date_attrs(train_data_r1w)
  dev_data_r2 = dummy_var_all_date_attrs(dev_data_r1w)
  test_data_r2 = dummy_var_all_date_attrs(test_data_r1w)

  LR = LinearRegression()

  #model 2, with t2 labels
  LR.fit(train_data_r2, train_labels_t1[:,2])
  log_pred_labels = LR.predict(dev_data_r2)

  print "model 2:\t" + str(LR.score(dev_data_r2, dev_labels_t1[:,2]))
  print RMSLE(log_pred_labels, dev_labels_t1[:,2])
  #plt.hist(log_pred_labels,bins=50)
  #plt.show()
  log_pred_labels = np.array(log_pred_labels)

  plt.scatter((np.exp(log_pred_labels) - 1), dev_labels[:,2], marker = '.')
  plt.ylabel("actual")
  plt.xlabel("predicted")
  #plt.xlim(0, 1000)
  #plt.ylim(0, 1000)
  remove_border()
  plt.show()

  #2
  LR.fit(np.vstack((train_data_r2, dev_data_r2)), np.append(train_labels_t1[:,2],dev_labels_t1[:,2]))
  log_pred_labels2 = LR.predict(test_data_r2)

  pred_labels2 = np.exp(log_pred_labels2) - 1
  print pred_labels2.shape

  #write_submission(pred_labels2)


  #regression of count and registered separately
  LR = LinearRegression()

  LR.fit(train_data_r2, train_labels_t1[:,0])
  log_pred_labels = LR.predict(dev_data_r2)

  print "model 1 - casual:\t" + str(LR.score(dev_data_r2, dev_labels_t1[:,0]))
  print RMSLE(log_pred_labels, dev_labels_t1[:,0])


  LR.fit(train_data_r2, train_labels_t1[:,1])
  log_pred_labels = LR.predict(dev_data_r2)

  print "model 1 - registered:\t" + str(LR.score(dev_data_r2, dev_labels_t1[:,1]))
  print RMSLE(log_pred_labels, dev_labels_t1[:,1])

  #Note the problem with the first histogram: the model is predicting some values of 
  #log(casual+1) are negative, meaning when we exponentiate them and subtract 1, they will be negative. 
  #i rectify this by setting a min of zero.


  LR.fit(np.vstack((train_data_r2,dev_data_r2)), np.append(train_labels_t1[:,0],dev_labels_t1[:,0]))
  log_pred_labels1_1 = LR.predict(test_data_r2)

  LR.fit(np.vstack((train_data_r2,dev_data_r2)), np.append(train_labels_t1[:,1],dev_labels_t1[:,1]))
  log_pred_labels1_2 = LR.predict(test_data_r2)

  casual_pred = np.exp(log_pred_labels1_1) - 1
  registered_pred = np.exp(log_pred_labels1_2) - 1

  casual_pred_t = casual_pred
  for i in range(len(casual_pred)):
    if casual_pred[i] < 0:
      casual_pred_t[i] = 0

  linear_regression_pred_labels1 = casual_pred_t + registered_pred
  #pred_labels1 = np.around(casual_pred_t) + np.around(np.exp(log_pred_labels1_2) - 1)  #round cas and reg
  #pred_labels1 = np.around(casual_pred_t + np.exp(log_pred_labels1_2) - 1) #round total
  print linear_regression_pred_labels1.shape


  #write_submission(pred_labels1)


## random forest

### data prep

#### functions

# In[ ]:

def add_dayofweek_as_single_attr(data):
  """
  add the day of the week as a single attribute
  """
  return_data = np.array([])
  for row in data:
    w = datetime.date(int(row[0]), int(row[1]), int(row[2])).weekday()
    return_row = np.append(row, w)
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data


def holiday_and_workingday_into_daytype(data):
  """
  combine holiday and workingday into variable, daytype
  """
  return_data = np.array([])
  for row in data:
    daytype = 0
    if row[6] == 1:
      daytype = 0
    elif row[5] == 0:
      daytype = 1
    else:
      daytype = 2
    return_row = np.append(row[0:5], row[7:])
    return_row = np.append(return_row, daytype)
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data


def remove_dayofmonth(data):
  """
  remove day of the month from the list of features
  """
  return_data = np.array([])
  for row in data:
    return_row = np.append(row[0:2], row[3:])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data


def remove_month(data):
  """
  remove month from the list of features
  """
  return_data = np.array([])
  for row in data:
    return_row = np.append(row[0:1], row[2:])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def remove_feature(data, feature):
  """
  remove feature from the list of features
  """
  return_data = np.array([])
  for row in data:
    return_row = np.append(row[0:feature-1], row[feature:])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data


def build_yester_dict_outcomes(h, arima_dict, data, labels):
  """
  build a dictionary of the outcome from h hours ago
  """
  arima_dict = {}
  for n in range(len(data)):
    arima_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) + datetime.timedelta(hours = h))] = labels[n,0:2]
  return arima_dict


def build_ma_dict_outcomes(h, ma_dict, yester_dict, data, labels):
  """
  build a dictionary of moving average of last h hours
  """
  ma_dict = {}
  for n in range(len(data)):
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) - datetime.timedelta(hours = h_i - 1)) for h_i in range(h)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in yester_dict:
        a += yester_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last h hours is missing, use current values as moving average of last h hours
      try:
        ma = yester_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])))]
      except:
        ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3]))+ datetime.timedelta(hours = 1))] = ma
  return ma_dict


def build_yester_dict_features(h, i, arima_dict, data):
  """
  build a dictionary of the outcome from h hours ago
  """
  arima_dict = {}
  for n in range(len(data)):
    arima_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) + datetime.timedelta(hours = h))] = data[n][i]
  return arima_dict


def build_ma_dict_features(h, ma_dict, yester_dict, data):
  """
  build a dictionary of moving average of last h hours
  """
  ma_dict = {}
  for n in range(len(data)):
    ma = [0.0]
    ma_date_list = [str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) - datetime.timedelta(hours = h_i-1)) for h_i in range(h)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in yester_dict:
        a += yester_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last h hours is missing, use current values as moving average of last h hours
      try:
        ma = yester_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])))]
      except:
        ma = [0.0]
    ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3]))+ datetime.timedelta(hours = h))] = ma
  return ma_dict

def add_yesterhour(data, yester_dict):
  """
  Use t-1 outcomes as predictor variables
  """
  return_data = np.array([])
  for n in range(len(data)):
    try:
      return_row = np.append(data[n], yester_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])))])
    except:
      return_row = np.append(data[n], [0,0])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def add_yesterhour_features(data, yester_dict):
  """
  Use t-1 outcomes as predictor variables
  """
  return_data = np.array([])
  for n in range(len(data)):
    try:
      return_row = np.append(data[n], yester_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])))])
    except:
      return_row = np.append(data[n], [0])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def add_ma(data, ma_dict):
  """
  moving average of outcomes (or features) as predictor variables
  """
  return_data = np.array([])
  for n in range(len(data)):
    try:
      return_row = np.append(data[n], ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])))])
    except:
      return_row = np.append(data[n], [0.0, 0.0])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def add_ma_features(data, ma_dict):
  """
  moving average of outcomes (or features) as predictor variables
  """
  return_data = np.array([])
  for n in range(len(data)):
    try:
      return_row = np.append(data[n], ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])))])
    except:
      return_row = np.append(data[n], [0.0])
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def eval_function(traindata, trainlabel, testdata, colnames=[], n_est = [1, 5, 10, 50, 100], print_features = False, print_plot = True):
  print "n\t\tc\tr"
  for n in n_est:
    rfra = RandomForestRegressor(n_estimators=n)
    rfra.fit(traindata, trainlabel)

    rfra_log_pred_labels = rfra.predict(testdata)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(rfra_log_pred_labels[:,0], dev_labels_t1[:,0]), RMSLE(rfra_log_pred_labels[:,1], dev_labels_t1[:,1]))
    if n == n_est[-1] and print_features == True:
      print "\nfeature importances:"
      for name, feature in zip(colnames, rfra.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)
  
  if print_plot:
    act_vs_pred_plot(np.append(np.exp(rfra_log_pred_labels[:,0])-1, np.exp(rfra_log_pred_labels[:,1])-1),np.append(dev_labels[:,0],dev_labels[:,1]))

def eval_function_single(traindata, trainlabel, testdata, colnames=[], n_est = [1, 5, 10, 50, 100], print_features = False, print_plot = True, model = 'c'):
  print "n\t\t"+model
  for n in n_est:
    rfra = RandomForestRegressor(n_estimators=n)
    rfra.fit(traindata, trainlabel)

    rfra_log_pred_labels = rfra.predict(testdata)
    
    if model == 'c':
      m = 0
    else:
      m = 1
    print "%8.3f\t%7.5f\t" % (n, RMSLE(rfra_log_pred_labels[:,m], dev_labels_t1[:,m]))
    if n == n_est[-1] and print_features == True:
      print "\nfeature importances:"
      for name, feature in zip(colnames, rfra.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)
  
  if print_plot:
    act_vs_pred_plot(np.exp(rfra_log_pred_labels[:,m])-1,dev_labels[:,m])

    
def act_vs_pred_plot(pred_labels, act_labels):
  plt.scatter(pred_labels, act_labels, marker = '.')
  plt.ylabel("actual")
  plt.xlabel("predicted")
  plt.xlim(0, 1000)
  plt.ylim(0, 1000)
  remove_border()
  plt.show()

  
  
def build_day_dict(day_dict, feature, data, cond = 0):
  """
  build dictionary that counts number of consecutive days with same value of feature. 
  for example, if two consecutive days both have values workingday = 1, day_dict would have a 2 for each of those days
  """
  d_dict = {}
  day_arr, feature_arr = [], []
  for n in range(len(data)):
    if datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2])) not in d_dict:
      d_dict[datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]))] = data[n][feature]

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


def build_day_feature(day_dict, train_data, dev_data, test_data):
  """
  append new feature based on day_dict to data
  """
  train_data_r = np.zeros((train_data.shape[0], train_data.shape[1]+1))
  dev_data_r = np.zeros((dev_data.shape[0], dev_data.shape[1]+1))
  test_data_r = np.zeros((test_data.shape[0], test_data.shape[1]+1))
  
  for i in range(len(train_data)):
    train_data_r[i] = np.append(train_data[i], day_dict[str(datetime.datetime(int(train_data[i][0]), int(train_data[i][1]), int(train_data[i][2])))])

  for i in range(len(dev_data)):
    dev_data_r[i] = np.append(dev_data[i], day_dict[str(datetime.datetime(int(dev_data[i][0]), int(dev_data[i][1]), int(dev_data[i][2])))])

  for i in range(len(test_data)):
    test_data_r[i] = np.append(test_data[i], day_dict[str(datetime.datetime(int(test_data[i][0]), int(test_data[i][1]), int(test_data[i][2])))])
  
  return train_data_r, dev_data_r, test_data_r



def build_past_period_dict(p_dict, period, feature, threshold, data):
  """
  build dictionary that counts number of occurances of a certain value in the past x hours.
  for example, count number of hours with weather > 2 in the last 12 hours.
  """
  return_dict = {}
  
  for n in range(len(data)):
    if datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])) not in p_dict:
      p_dict[datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3]))] = data[n][feature]

  ordered_dict = collections.OrderedDict(sorted(p_dict.items()))
  
  for key, value in ordered_dict.iteritems():
    date_list = [key - datetime.timedelta(hours = h_i+1) for h_i in range(period)]
    
    count = 0
    for d in date_list:
      if d in ordered_dict and ordered_dict[d] > threshold:
        count += 1
    
    return_dict[key] = count
        
  return return_dict


def build_period_feature(period_dict, train_data, dev_data, test_data):
  """
  append new feature based on period_dict to data
  """
  train_data_r = np.zeros((train_data.shape[0], train_data.shape[1]+1))
  dev_data_r = np.zeros((dev_data.shape[0], dev_data.shape[1]+1))
  test_data_r = np.zeros((test_data.shape[0], test_data.shape[1]+1))
  
  for i in range(len(train_data)):
    train_data_r[i] = np.append(train_data[i], period_dict[datetime.datetime(int(train_data[i][0]), int(train_data[i][1]), int(train_data[i][2]), int(train_data[i][3]))])

  for i in range(len(dev_data)):
    dev_data_r[i] = np.append(dev_data[i], period_dict[datetime.datetime(int(dev_data[i][0]), int(dev_data[i][1]), int(dev_data[i][2]), int(dev_data[i][3]))])

  for i in range(len(test_data)):
    test_data_r[i] = np.append(test_data[i], period_dict[datetime.datetime(int(test_data[i][0]), int(test_data[i][1]), int(test_data[i][2]), int(test_data[i][3]))])
  
  return train_data_r, dev_data_r, test_data_r


def build_yester_dict_outcomes_full(h, arima_dict, data, labels):
  """
  build a dictionary of the outcome from h hours ago
  """
  arima_dict = {}
  for n in range(len(data)):
    arima_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) + datetime.timedelta(hours = h))] = labels[n,:]
  return arima_dict


def build_ma_dict_outcomes_full(h, ma_dict, yester_dict, data, labels, default = [36.022, 155.552, 191.574]):
  """
  build a dictionary of moving average of last h hours
  """
  ma_dict = {}
  for n in range(len(data)):
    ma = default
    ma_date_list = [str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3])) - datetime.timedelta(hours = h_i - 1)) for h_i in range(h)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in yester_dict:
        a += yester_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      ma = default
    ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int (data[n][3]))+ datetime.timedelta(hours = 1))] = ma
  return ma_dict


def add_ma_full(data, ma_dict, default = [36.022, 155.552, 191.574]):
  """
  moving average of outcomes as predictor variables
  default = [36.022, 155.552, 191.574] is the average of all outcomes we have
  """
  return_data = np.array([])
  for n in range(len(data)):
    try:
      return_row = np.append(data[n], ma_dict[str(datetime.datetime(int(data[n][0]), int(data[n][1]), int(data[n][2]), int(data[n][3])))])
    except:
      return_row = np.append(data[n], default)
    try:
      return_data = np.vstack((return_data, return_row))
    except:
      return_data = return_row
  return return_data

def recode_feature(data, feature, recode_map):
  #for season, recode_map is [0, 3, 1, 2]
  #for month, recode map is
  return_data = np.zeros_like(data)
  for i in range(len(data)):
    for j in range(len(recode_map)):
      if data[i][feature] == recode_map[j]:
        r = data[i]
        r[feature] = j
        return_data[i] = r
        break
  return return_data

def build_recode_map(data, feature):
  recode_map = {}
  
  cat_set = []
  for i in data[:,feature]:
    if int(i) not in cat_set:
      cat_set.append(int(i))
  
  #cat_set = [i+1 for i in range(12)]
  bar = []
  for i in cat_set:
    s = 0
    cnt = 0
    for di in range(len(data[:,feature])):
      if int(data[di,feature]) == i:
        cnt += 1.
        s += labels[di,0]
    bar.append(s/cnt)

  dic = {}

  for i, j in zip(cat_set, bar):
    dic[j] = i

  o_dic = collections.OrderedDict(sorted(dic.items()))

  recode_map = []
  for i,j in o_dic.iteritems():
    recode_map.append(j)
  return recode_map

def humid_less_than_16(train_data, dev_data, test_data):
  train_data_r = np.zeros((train_data.shape[0], train_data.shape[1]+1))
  dev_data_r = np.zeros((dev_data.shape[0], dev_data.shape[1]+1))
  test_data_r = np.zeros((test_data.shape[0], test_data.shape[1]+1))
  
  n = 0
  
  for i in range(len(train_data)):
    if train_data[i][10] < 16:
      n = 1
    else:
      n = 0
    train_data_r[i] = np.append(train_data[i], n)

  for i in range(len(dev_data)):
    if dev_data[i][10] < 16:
      n = 1
    else:
      n = 0
    dev_data_r[i] = np.append(dev_data[i], n)

  for i in range(len(test_data)):
    if test_data[i][10] < 16:
      n = 1
    else:
      n = 0
    test_data_r[i] = np.append(test_data[i], n)
  
  return train_data_r, dev_data_r, test_data_r

def add_weekday_hour_avg_to_array(array_data, weekday_hour_avg_matrix):
    new_column = np.zeros((len(array_data),1))
    for row_i in range(len(array_data)):
        new_column[row_i,0] = weekday_hour_avg_matrix[array_data[row_i,12],array_data[row_i,3]]
    array_data = np.hstack((array_data, new_column))  
    return array_data

def add_weekday_hour_avg(train_data, dev_data, test_data, column_names, train_labels, dev_labels):
    column_names = column_names + ['weekday_hour_avg']
    data = np.vstack((train_data, dev_data))
    labels =np.vstack((train_labels, dev_labels))
    weekday_hour_avg_matrix = np.zeros((7,24))
    for weekday in range(7):
        for hour in range(24):
            weekday_hour_labels = labels[(data[:,12] == weekday) & 
                                               (data[:,3] == hour)][:,2]
            weekday_hour_avg_matrix[weekday,hour] = weekday_hour_labels.mean()
    
    train_new = add_weekday_hour_avg_to_array(train_data, weekday_hour_avg_matrix)
    dev_new = add_weekday_hour_avg_to_array(dev_data, weekday_hour_avg_matrix)
    test_new = add_weekday_hour_avg_to_array(test_data, weekday_hour_avg_matrix)
    
    return train_new, dev_new, test_new, column_names

def atemp_outlier_fix(dev_data):
  """
  fix outliers for atemp
  they all fall on august 17, 2012
  """
  dev_data_new = dev_data
  for i in range(len(dev_data)):
    if dev_data[i][8]>20 and dev_data[i][9]==12.12:
      dev_data_new[i][9] = dev_data[i][8]
  
  return dev_data_new


# In[ ]:
def determine_output_averages():
  #what is the average of all counts
  ##so that we can add these (instead of zeros) when previous hours are missing
  ac = 0.
  ar = 0.

  for row in np.vstack((train_labels, dev_labels)):
    ac += row[0]
    ar += row[1]
    
  ac /= (train_labels.shape[0]+dev_labels.shape[0])
  ar /= (train_labels.shape[0]+dev_labels.shape[0])

  print ac
  print ar
  print ac+ar

def rfr_feature_explorations():
  # I tried recoding season, month, and hour to be in order of average outcome. this made the model worse for any combination of these changes. i am not sure why, this doesnt make sense. i expected either no change or better, not worse.
  # ###implemented a feature that detects 3-day+ weekends. it improved the score!
  # ###added a feature to see if there is are wet grounds from rain/snow from the last 3 hours. it improved the score!
  # tried adding a dummy variable for when humidity is less than 16. did not get an improvement and feature importance was extremely low.
  # ###some features hurt the model for casual, but help registered, and vice versa (and some help both of course). multi-output form for rfr is convenient but it forces the use of same features for both models. i will attempt to separate the models.

  # In[ ]:

  #last of the features
  n_est = [100]

  #eval_function_single(train_data, train_labels_t1[:,0:2], dev_data, column_names, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data, train_labels_t1[:,0:2], dev_data, column_names, n_est, print_features=True, print_plot=False, model = 'r')

  dev_data = atemp_outlier_fix(dev_data)

  #add day of week as a single attribute
  ##improves both
  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'r')

  #add if a day belongs to x_day_weekend
  ##improves both
  day_dict = build_day_dict({}, 6, np.vstack((train_data, dev_data, test_data)))

  train_data_rf, dev_data_rf, test_data_rf = build_day_feature(day_dict, train_data_rf, dev_data_rf, test_data_rf)
  column_names_rf = column_names_rf + ['non_wk_dy_cnt']
  print train_data_rf.shape

  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'r')

  #eval_function(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False)

  #add number of hours the ground was wet in the last x hours
  #try 1, 2, 3, 4, 5, 6, 12
  #3 is best
  ##improves both
  p_dict3 = build_past_period_dict({}, 3, 7, 2, np.vstack((train_data, dev_data, test_data)))

  train_data_rf, dev_data_rf, test_data_rf = build_period_feature(p_dict3, train_data_rf, dev_data_rf, test_data_rf)
  column_names_rf = column_names_rf + ['wet_grounds']
  print train_data_rf.shape

  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'r')

  #eval_function(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False)

  #add weekday hour average

  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = add_weekday_hour_avg(train_data_rf, dev_data_rf, test_data_rf, column_names_rf, train_labels, dev_labels)
  print train_data_rf.shape

  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'r')


  #add temp from last hour
  ##improves both
  ydict = {}
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rfc = add_yesterhour_features(train_data_rf, yester_dict_temp)
  dev_data_rfc = add_yesterhour_features(dev_data_rf, yester_dict_temp)
  test_data_rfc = add_yesterhour_features(test_data_rf, yester_dict_temp)

  column_names_rfc = column_names_rf + ['last_temp']
  print train_data_rfc.shape

  #eval_function_single(train_data_rfc, train_labels_t1[:,0:2], dev_data_rfc, column_names_rf, n_est, print_features=True, print_plot=False, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False, model = 'r')

  #add atemp from last hour
  ##help casual but not registered
  ##split and rename to indicate which model the data is for
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rfc = add_yesterhour_features(train_data_rfc, yester_dict_atemp)
  dev_data_rfc = add_yesterhour_features(dev_data_rfc, yester_dict_atemp)
  test_data_rfc = add_yesterhour_features(test_data_rfc, yester_dict_atemp)

  column_names_rfc = column_names_rfc + ['last_atemp']
  print train_data_rf.shape

  eval_function_single(train_data_rfc, train_labels_t1[:,0:2], dev_data_rfc, column_names_rfc, n_est, print_features=True, print_plot=True, model = 'c')
  eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=True, model = 'r')

  #eval_function(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False)


  eval_function_single(train_data, train_labels_t1[:,0:2], dev_data, column_names, n_est, print_features=True, print_plot=True, model = 'c')
  eval_function_single(train_data, train_labels_t1[:,0:2], dev_data, column_names, n_est, print_features=True, print_plot=True, model = 'r')


  #remove day of month
  ##this seems to hurt both
  #train_data_rf = remove_feature(train_data_rf, 2)
  #dev_data_rf = remove_feature(dev_data_rf, 2)
  #test_data_rf = remove_feature(test_data_rf, 2)
  #column_names_rf = column_names_rf[:2] + column_names_rf[3:]
  #print train_data_rf.shape

  #train_data_rfc = remove_feature(train_data_rfc, 2)
  #dev_data_rfc = remove_feature(dev_data_rfc, 2)
  #test_data_rfc = remove_feature(test_data_rfc, 2)
  #column_names_rfc = column_names_rfc[:2] + column_names_rfc[3:]
  #print train_data_rfc.shape

  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=True, model = 'c')
  #eval_function_single(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=True, model = 'r')

  #num_bins = 50
  #plt.hist(np.append(train_data_rf[:,-1],dev_data_rf[:,-1]),numBins,color='green',alpha=0.8)
  #plt.hist(test_data_rfp[:,-1],numBins,color='red',alpha=0.6)
  #plt.show()


# In[ ]:

def data_prep(train_data, dev_data, test_data, column_names):
  
  dev_data = atemp_outlier_fix(dev_data)
  
  #add day of week as a single attribute
  ##improves both
  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #add if a day belongs to x_day_weekend
  ##improves both
  day_dict = build_day_dict({}, 6, np.vstack((train_data, dev_data, test_data)))

  train_data_rf, dev_data_rf, test_data_rf = build_day_feature(day_dict, train_data_rf, dev_data_rf, test_data_rf)
  column_names_rf = column_names_rf + ['non_wk_dy_cnt']
  print train_data_rf.shape

  #add number of hours the ground was wet in the last 3 hours
  ##improves both
  p_dict3 = build_past_period_dict({}, 3, 7, 2, np.vstack((train_data, dev_data, test_data)))

  train_data_rf, dev_data_rf, test_data_rf = build_period_feature(p_dict3, train_data_rf, dev_data_rf, test_data_rf)
  column_names_rf = column_names_rf + ['wet_grounds']
  print train_data_rf.shape

  #add weekly hours average
  ##improves both
  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = add_weekday_hour_avg(train_data_rf, dev_data_rf, test_data_rf, column_names_rf, train_labels, dev_labels)
  print train_data_rf.shape
  
  #add temp from last hour
  ##helps casual
  ydict = {}
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rfc = add_yesterhour_features(train_data_rf, yester_dict_temp)
  dev_data_rfc = add_yesterhour_features(dev_data_rf, yester_dict_temp)
  test_data_rfc = add_yesterhour_features(test_data_rf, yester_dict_temp)

  column_names_rfc = column_names_rf + ['last_temp']
  print train_data_rfc.shape

  #add atemp from last hour
  ##help casual but not registered
  ##split and rename to indicate which model the data is for
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rfc = add_yesterhour_features(train_data_rfc, yester_dict_atemp)
  dev_data_rfc = add_yesterhour_features(dev_data_rfc, yester_dict_atemp)
  test_data_rfc = add_yesterhour_features(test_data_rfc, yester_dict_atemp)

  column_names_rfc = column_names_rf + ['last_atemp']
  print train_data_rf.shape

  return train_data_rfc, dev_data_rfc, test_data_rfc, column_names_rfc, train_data_rf, dev_data_rf, test_data_rf, column_names_rf

def add_ma_to_data(data, ma_dict, l):
  datama = np.zeros((data.shape[0], data.shape[1]+l))

  for i in range(len(data)):
    datama[i] = np.append(data[i], ma_dict[str(datetime.datetime(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])))])
  
  return datama

def rfr_test_predict(combined_data, combined_labels, test_data, n_est):
  rfr = RandomForestRegressor(n_estimators=n_est)
  rfr.fit(combined_data, combined_labels)
  pred_labels = rfr.predict(test_data)
  
  print 'Predicted test labels shape:', pred_labels.shape
  return pred_labels

def rfr_2pass_submission():
  ##prep data
  train_data_rfc, dev_data_rfc, test_data_rfc, column_names_rfc, train_data_rf, dev_data_rf, test_data_rf, column_names_rf = data_prep(train_data, dev_data, test_data, column_names)

  # Execute RFR 
  print_div_line('RFR - pass 1')
  test_log_labels_pass1c = rfr_test_predict(np.vstack((train_data_rfc, dev_data_rfc)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_rfc, 300)
  test_log_labels_pass1r = rfr_test_predict(np.vstack((train_data_rf, dev_data_rf)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_rf, 300)
  print


  # Reinit
  train_data_rfc, dev_data_rfc, test_data_rfc, column_names_rfc, train_data_rf, dev_data_rf, test_data_rf, column_names_rf = data_prep(train_data, dev_data, test_data, column_names)


  ###Prepare data with moving average features

  test_labels_pass1c = np.exp(test_log_labels_pass1c) - 1
  test_labels_pass1r = np.exp(test_log_labels_pass1r) - 1
  test_labels_pass1 = np.vstack((test_labels_pass1c[:,0], test_labels_pass1r[:,1], test_labels_pass1c[:,0]+test_labels_pass1c[:,1])).T
  print test_labels_pass1.shape
          

  #find ma of last 2 hours and add to data
  test_set_c = np.zeros((test_data_rfc.shape[0], 7))
  for i in range(len(test_data_rfc)):
    test_set_c[i] = np.append(test_data_rfc[i,:4], test_labels_pass1[i])

  train_set_c = np.zeros((np.vstack((train_data_rfc, dev_data_rfc)).shape[0], 7))
  for i in range(len(np.vstack((train_data_rfc, dev_data_rfc)))):
    train_set_c[i] = np.append(np.vstack((train_data_rfc, dev_data_rfc))[i,:4], np.vstack((train_labels, dev_labels))[i])


  test_set_r = np.zeros((test_data_rf.shape[0], 7))
  for i in range(len(test_data_rf)):
    test_set_r[i] = np.append(test_data_rf[i,:4], test_labels_pass1[i])

  train_set_r = np.zeros((np.vstack((train_data_rf, dev_data_rf)).shape[0], 7))
  for i in range(len(np.vstack((train_data_rf, dev_data_rf)))):
    train_set_r[i] = np.append(np.vstack((train_data_rf, dev_data_rf))[i,:4], np.vstack((train_labels,dev_labels))[i])

    
  total_set_c = np.vstack((train_set_c, test_set_c))  
  total_set_r = np.vstack((train_set_r, test_set_r))


  default = [36.022, 155.552, 191.574]

  label_dict = {}
  ma_dict_c = {}
  for n in range(len(total_set_c)):
    label_dict[str(datetime.datetime(int(total_set_c[n][0]), int(total_set_c[n][1]), int(total_set_c[n][2]), int(total_set_c[n][3])))] = total_set_c[n, 4:]

  for n in range(len(total_set_c)):
    ma = default
    ma_date_list = [str(datetime.datetime(int(total_set_c[n][0]), int(total_set_c[n][1]), int(total_set_c[n][2]), int(total_set_c[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      ma = default
    ma_dict_c[str(datetime.datetime(int(total_set_c[n][0]), int(total_set_c[n][1]), int(total_set_c[n][2]), int(total_set_c[n][3])))] = ma

    
  label_dict = {}
  ma_dict_r = {}
  for n in range(len(total_set_r)):
    label_dict[str(datetime.datetime(int(total_set_r[n][0]), int(total_set_r[n][1]), int(total_set_r[n][2]), int(total_set_r[n][3])))] = total_set_r[n, 4:]

  for n in range(len(total_set_r)):
    ma = default
    ma_date_list = [str(datetime.datetime(int(total_set_r[n][0]), int(total_set_r[n][1]), int(total_set_r[n][2]), int(total_set_r[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      ma = default
    ma_dict_r[str(datetime.datetime(int(total_set_c[n][0]), int(total_set_c[n][1]), int(total_set_c[n][2]), int(total_set_c[n][3])))] = ma
    
  train_data_pass2r = add_ma_to_data(train_data_rf, ma_dict_r, 3)
  dev_data_pass2r = add_ma_to_data(dev_data_rf, ma_dict_r, 3)
  test_data_pass2r = add_ma_to_data(test_data_rf, ma_dict_r, 3)

  train_data_pass2c = add_ma_to_data(train_data_rfc, ma_dict_c, 3)
  dev_data_pass2c = add_ma_to_data(dev_data_rfc, ma_dict_c, 3)
  test_data_pass2c = add_ma_to_data(test_data_rfc, ma_dict_c, 3)

  ##

  #remove day of month
  train_data_pass2r = remove_feature(train_data_pass2r, 2)
  dev_data_pass2r = remove_feature(dev_data_pass2r, 2)
  test_data_pass2r = remove_feature(test_data_pass2r, 2)

  train_data_pass2c = remove_feature(train_data_pass2c, 2)
  dev_data_pass2c = remove_feature(dev_data_pass2c, 2)
  test_data_pass2c = remove_feature(test_data_pass2c, 2)

  #column_names_pass2 = np.append(column_names_rf[:2], column_names_rf[3:])
  #print train_data_pass2.shape

  print 'Test data shape: ', test_data_pass2c.shape
  print 'Test data shape: ', test_data_pass2r.shape

  # Execute RFR with predicted test_labels in the test_data set
  print_div_line('RFR - pass2')

  test_log_labels_pass2c = rfr_test_predict(np.vstack((train_data_pass2c, dev_data_pass2c)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2c, 300)
  test_log_labels_pass2r = rfr_test_predict(np.vstack((train_data_pass2r, dev_data_pass2r)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2r, 300)
  print

  #print RMSLE(test_log_labels_pass2c[:,0],dev_labels_t1[:,0])
  #print RMSLE(test_log_labels_pass2c[:,1],dev_labels_t1[:,1])


  # Prepare submission
  count_pred_c = np.exp(test_log_labels_pass2c[:,0]) - 1
  count_pred_r = np.exp(test_log_labels_pass2c[:,1]) - 1
  count_pred = count_pred_c + count_pred_r
  print count_pred.shape

  print "%d predicted count values <= 0" % len(count_pred[count_pred<0])

  for i in range(len(count_pred)):
    if count_pred[i] < 0:
      count_pred[i] = 0

  #write_submission(count_pred)

def features_discarded():
  # In[ ]:

  #discarded:

  #try recoding month
  month_recode = build_recode_map(data, 3)

  train_data_rf = recode_feature(train_data_rf, 3, month_recode)
  dev_data_rf = recode_feature(dev_data_rf, 3, month_recode)
  test_data_rf = recode_feature(test_data_rf, 3, month_recode)
  column_names_rf[1] = 're_month'
  print train_data_rf.shape

  cat_vars = [1]

  data = np.vstack((train_data_rf, dev_data_rf))

  for c in cat_vars:
    cat_set = []
    for i in np.vstack((train_data_rf,dev_data_rf))[:,c]:
      if int(i) not in cat_set:
        cat_set.append(int(i))

    bar_h0 = []
    bar_h1 = []
    bar_hmissing = []
    for i in cat_set:
      s0 = 0
      s1 = 0
      cnt = 0
      for di in range(len(data[:,c])):
        if int(data[di,c]) == i:
          cnt += 1
          s0 += labels[di,0]
          s1 += labels[di,1]
      if cnt > 0: 
        bar_h0.append(s0/cnt)
        bar_h1.append(s1/cnt)
        bar_hmissing.append(0)
      else:
        bar_h0.append(0)
        bar_h1.append(0)
        bar_hmissing.append(10)

    plt.bar(cat_set, bar_h0, color = colors[0])
    plt.bar(cat_set, bar_h1, bottom = bar_h0, color = colors[1])
    plt.bar(cat_set, bar_hmissing, color = 'red')

    
  #add dummy variable for humidity < 16
  train_data_rf, dev_data_rf, test_data_rf = humid_less_than_15(train_data_rf, dev_data_rf, test_data_rf)
  column_names_rf = column_names_rf + ['humid_16']
  print train_data_rf.shape

  eval_function(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True, print_plot=False)

def feature_engineering_2():
  ### feature engineering

  # In[ ]:

  #add day of week as a single attribute

  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #combine holiday and workingday into daytype

  train_data_rf = holiday_and_workingday_into_daytype(train_data_rf)
  dev_data_rf = holiday_and_workingday_into_daytype(dev_data_rf)
  test_data_rf = holiday_and_workingday_into_daytype(test_data_rf)
  column_names_rf = column_names_rf[0:5] + column_names_rf[7:] + ['daytype']
  print train_data_rf.shape

  ##test three sets below
  #remove month
  train_data_rf1 = remove_month(train_data_rf)
  dev_data_rf1 = remove_month(dev_data_rf)
  test_data_rf1 = remove_month(test_data_rf)
  column_names_rf1 = column_names_rf[:1] + column_names_rf[2:]
  print train_data_rf1.shape

  #remove day of month
  train_data_rf2 = remove_dayofmonth(train_data_rf)
  dev_data_rf2 = remove_dayofmonth(dev_data_rf)
  test_data_rf2 = remove_dayofmonth(test_data_rf)
  column_names_rf2 = column_names_rf[:2] + column_names_rf[3:]
  print train_data_rf2.shape

  #remove both
  train_data_rf3 = remove_dayofmonth(train_data_rf1)
  dev_data_rf3 = remove_dayofmonth(dev_data_rf1)
  test_data_rf3 = remove_dayofmonth(test_data_rf1)
  column_names_rf3 = column_names_rf1[:1] + column_names_rf1[2:]
  print train_data_rf3.shape



  # In[ ]:

  n_est = [5, 10, 50, 100]

  eval_function(train_data_rf1, train_labels_t1[:,0:2], dev_data_rf1, column_names_rf1, n_est)
  eval_function(train_data_rf2, train_labels_t1[:,0:2], dev_data_rf2, column_names_rf2, n_est)
  eval_function(train_data_rf3, train_labels_t1[:,0:2], dev_data_rf3, column_names_rf3, n_est)


  ## removing day of month is more effective that removing month, or removing both

  # In[ ]:

  #restart
  #add day of week as a single attribute

  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #combine holiday and workingday into daytype

  train_data_rf = holiday_and_workingday_into_daytype(train_data_rf)
  dev_data_rf = holiday_and_workingday_into_daytype(dev_data_rf)
  test_data_rf = holiday_and_workingday_into_daytype(test_data_rf)
  column_names_rf = column_names_rf[0:5] + column_names_rf[7:] + ['daytype']
  print train_data_rf.shape

  #remove day of month
  train_data_rf = remove_dayofmonth(train_data_rf)
  dev_data_rf = remove_dayofmonth(dev_data_rf)
  test_data_rf = remove_dayofmonth(test_data_rf)
  column_names_rf = column_names_rf[:2] + column_names_rf[3:]
  print train_data_rf.shape

  #add ma of outcomes of last 2 hours
  ydict = {}
  yester_dict = build_yester_dict_outcomes(1, ydict, np.vstack((train_data, dev_data)), np.vstack((train_labels, dev_labels)))
  mdict = {}
  ma_dict = build_ma_dict_outcomes(2, mdict, yester_dict, np.vstack((train_data, dev_data)), np.vstack((train_labels, dev_labels)))

  train_data_rf = add_ma(train_data_rf, ma_dict)
  dev_data_rf = add_ma(dev_data_rf, ma_dict)
  column_names_rf = column_names_rf + ['cas_ma', 'reg_ma']
  print train_data_rf.shape

  n_est = [5, 10, 50, 100]
  eval_function(train_data_rf, train_labels_t1[:,0:2], dev_data_rf, column_names_rf, n_est, print_features=True)

  #try adding weather, temp, and atemp from last hour
  ydict = {}
  yester_dict1 = build_yester_dict_features(1, 7, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf1 = add_yesterhour_features(train_data_rf, yester_dict1)
  dev_data_rf1 = add_yesterhour_features(dev_data_rf, yester_dict1)
  column_names_rf1 = column_names_rf + ['last_wea']
  print train_data_rf1.shape
  print dev_data_rf1.shape

  eval_function(train_data_rf1, train_labels_t1[:,0:2], dev_data_rf1, column_names_rf1, n_est)


  yester_dict2 = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf2 = add_yesterhour_features(train_data_rf, yester_dict2)
  dev_data_rf2 = add_yesterhour_features(dev_data_rf, yester_dict2)
  column_names_rf2 = column_names_rf + ['last_temp']
  print train_data_rf2.shape
  print dev_data_rf2.shape

  eval_function(train_data_rf2, train_labels_t1[:,0:2], dev_data_rf2, column_names_rf2, n_est)


  yester_dict3 = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf3 = add_yesterhour_features(train_data_rf, yester_dict3)
  dev_data_rf3 = add_yesterhour_features(dev_data_rf, yester_dict3)
  column_names_rf3 = column_names_rf + ['last_atemp']
  print train_data_rf3.shape
  print dev_data_rf3.shape

  eval_function(train_data_rf3, train_labels_t1[:,0:2], dev_data_rf3, column_names_rf3, n_est)

  #try adding ma of weather, temp, atemp from last 2 hours
  mdict = {}
  ma_dict = build_ma_dict_features(2, mdict, yester_dict1, np.vstack((train_data, dev_data)))

  train_data_rf1m = add_ma_features(train_data_rf, yester_dict1)
  dev_data_rf1m = add_ma_features(dev_data_rf, yester_dict1)
  column_names_rf1m = column_names_rf + ['last_wea']
  print train_data_rf1m.shape
  print dev_data_rf1m.shape

  eval_function(train_data_rf1m, train_labels_t1[:,0:2], dev_data_rf1m, column_names_rf1m, n_est)

  ma_dict = build_ma_dict_features(2, mdict, yester_dict2, np.vstack((train_data, dev_data)))
  train_data_rf2m = add_ma_features(train_data_rf, yester_dict2)
  dev_data_rf2m = add_ma_features(dev_data_rf, yester_dict2)
  column_names_rf2m = column_names_rf + ['last_temp']
  print train_data_rf2m.shape
  print dev_data_rf2m.shape

  eval_function(train_data_rf2m, train_labels_t1[:,0:2], dev_data_rf2m, column_names_rf2, n_est)

  ma_dict = build_ma_dict_features(2, mdict, yester_dict3, np.vstack((train_data, dev_data)))
  train_data_rf3m = add_ma_features(train_data_rf, yester_dict3)
  dev_data_rf3m = add_ma_features(dev_data_rf, yester_dict3)
  column_names_rf3m = column_names_rf + ['last_atemp']
  print train_data_rf3m.shape
  print dev_data_rf3m.shape

  eval_function(train_data_rf3m, train_labels_t1[:,0:2], dev_data_rf3m, column_names_rf3, n_est)



  ### it appears temp and atemp are the better ones to add. single value seems better than ma.

  # In[ ]:

  #restart
  n_est = [30]

  #add day of week as a single attribute

  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #combine holiday and workingday into daytype

  train_data_rf = holiday_and_workingday_into_daytype(train_data_rf)
  dev_data_rf = holiday_and_workingday_into_daytype(dev_data_rf)
  test_data_rf = holiday_and_workingday_into_daytype(test_data_rf)
  column_names_rf = column_names_rf[0:5] + column_names_rf[7:] + ['daytype']
  print train_data_rf.shape

  #remove day of month
  train_data_rf = remove_dayofmonth(train_data_rf)
  dev_data_rf = remove_dayofmonth(dev_data_rf)
  test_data_rf = remove_dayofmonth(test_data_rf)
  column_names_rf = column_names_rf[:2] + column_names_rf[3:]
  print train_data_rf.shape

  #add temp from last hour
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf = add_yesterhour_features(train_data_rf, yester_dict_temp)
  dev_data_rf = add_yesterhour_features(dev_data_rf, yester_dict_temp)
  test_data_rf = add_yesterhour_features(test_data_rf, yester_dict_temp)

  column_names_rf = column_names_rf + ['last_temp']
  print train_data_rf.shape

  #add atemp from last hour
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf = add_yesterhour_features(train_data_rf, yester_dict_atemp)
  dev_data_rf = add_yesterhour_features(dev_data_rf, yester_dict_atemp)
  test_data_rf = add_yesterhour_features(test_data_rf, yester_dict_atemp)

  column_names_rf = column_names_rf + ['last_atemp']
  print train_data_rf.shape

  ##rename variables
  #add ma of outcomes of last 2 hours
  ydict = {}
  yester_dict = build_yester_dict_outcomes(1, ydict, np.vstack((train_data, dev_data)), np.vstack((train_labels, dev_labels)))
  mdict = {}
  ma_dict = build_ma_dict_outcomes(2, mdict, yester_dict, np.vstack((train_data, dev_data)), np.vstack((train_labels, dev_labels)))

  train_data_rfma = add_ma(train_data_rf, ma_dict)
  dev_data_rfma = add_ma(dev_data_rf, ma_dict)
  column_names_rfma = column_names_rf + ['cas_ma', 'reg_ma']
  print train_data_rfma.shape

  eval_function(train_data_rfma, train_labels_t1[:,0:2], dev_data_rfma, column_names_rfma, n_est, print_features=True)

def one_pass_rfr():
  ### one pass strategy (one at a time)

  # In[ ]:

  n_est = 300
  #random forest with arima
  rfra = RandomForestRegressor(n_estimators=n_est)
  rfra.fit(np.vstack((train_data_rfma, dev_data_rfma)), np.vstack((train_labels_t1[:,0:2],dev_labels_t1[:,0:2])))

  #regular random forest
  rfr = RandomForestRegressor(n_estimators=n_est)
  rfr.fit(np.vstack((train_data_rf, dev_data_rf)), np.vstack((train_labels_t1[:,0:2],dev_labels_t1[:,0:2])))


  rfra_log_pred_labels = np.array([])

  missing_cnt = 0

  for n in range(len(test_data)):
    # look at the last hour. 
    try:
      new_labels = rfra.predict(np.append(test_data_rf[n], ma_dict[str(datetime.datetime(int(test_data[n][0]), int(test_data[n][1]), int(test_data[n][2]), int(test_data[n][3])))]))
    #if missing, use normal rf model.
    except:
      missing_cnt += 1
      new_labels = rfr.predict(test_data_rf[n])

    yester_dict[str(datetime.datetime(int(test_data[n][0]), int(test_data[n][1]), int(test_data[n][2]), int(test_data[n][3])) + datetime.timedelta(hours=1))] = np.exp(new_labels) - 1
    
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(test_data[n][0]), int(test_data[n][1]), int(test_data[n][2]), int(test_data[n][3])) - datetime.timedelta(hours = h_i - 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in yester_dict:
        a += yester_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last h hours is missing, use current values as moving average of last h hours
      try:
        ma = yester_dict[str(datetime.datetime(int(test_data[n][0]), int(test_data[n][1]), int(test_data[n][2]), int(test_data[n][3])))]
      except:
        ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(test_data[n][0]), int(test_data[n][1]), int(test_data[n][2]), int(test_data[n][3]))+ datetime.timedelta(hours=1))] = ma
    
    try:
      rfra_log_pred_labels = np.vstack((rfra_log_pred_labels, new_labels)) 
    except:
      rfra_log_pred_labels = new_labels
       
  rfra_casual_pred = np.exp(rfra_log_pred_labels[:,0]) - 1
  rfra_registered_pred = np.exp(rfra_log_pred_labels[:,1]) - 1

  print "%d predicted casual values <= 0" % len(rfra_casual_pred[rfra_casual_pred<0])

  rfra_pred_labels = rfra_casual_pred + rfra_registered_pred

  print "%d predicted count values <= 0" % len(rfra_pred_labels[rfra_pred_labels<0])

  rfra_pred_labels_t2 = rfra_pred_labels
  for i in range(len(rfra_pred_labels)):
    if rfra_pred_labels[i] < 0:
      rfra_pred_labels_t[i] = 0

  print "data was missing %d times." % missing_cnt

  #write_submission(rfra_pred_labels_t2)


#### well that didnt work :(


# ###upon closer inspection of judd's code, i found something else we have been doing differently: judd has been feeding in all 3 outcomes to RFR (casual, registered, count) whereas i have only been feeding 2 (casual and registered) and then after getting the results back, adding them to get count.
# and for some reason (probably features) i cant get a score close to our best
# 

### Functions for multiple passes

# In[ ]:

def data_init():
  
  #add day of week as a single attribute
  train_data_rf = add_dayofweek_as_single_attr(train_data)
  dev_data_rf = add_dayofweek_as_single_attr(dev_data)
  test_data_rf = add_dayofweek_as_single_attr(test_data)
  column_names_rf = column_names + ['weekday']
  print train_data_rf.shape

  #combine holiday and workingday into daytype
  train_data_rf = holiday_and_workingday_into_daytype(train_data_rf)
  dev_data_rf = holiday_and_workingday_into_daytype(dev_data_rf)
  test_data_rf = holiday_and_workingday_into_daytype(test_data_rf)
  column_names_rf = column_names_rf[0:5] + column_names_rf[7:] + ['daytype']
  print train_data_rf.shape

  #add temp from last hour
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf = add_yesterhour_features(train_data_rf, yester_dict_temp)
  dev_data_rf = add_yesterhour_features(dev_data_rf, yester_dict_temp)
  test_data_rf = add_yesterhour_features(test_data_rf, yester_dict_temp)

  column_names_rf = column_names_rf + ['last_temp']
  print train_data_rf.shape

  #add atemp from last hour
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_rf = add_yesterhour_features(train_data_rf, yester_dict_atemp)
  dev_data_rf = add_yesterhour_features(dev_data_rf, yester_dict_atemp)
  test_data_rf = add_yesterhour_features(test_data_rf, yester_dict_atemp)

  column_names_rf = column_names_rf + ['last_atemp']
  print train_data_rf.shape
  
  return train_data_rf, dev_data_rf, test_data_rf, column_names_rf


def remove_dom(train_data_rf, dev_data_rf, test_data_rf, column_names_rf):

  #remove day of month
  train_data_rf = remove_dayofmonth(train_data_rf)
  dev_data_rf = remove_dayofmonth(dev_data_rf)
  test_data_rf = remove_dayofmonth(test_data_rf)
  column_names_rf = column_names_rf[:2] + column_names_rf[3:]
  print train_data_rf.shape
  
  return train_data_rf, dev_data_rf, test_data_rf, column_names_rf


def add_ma_to_data(data, ma_dict):
  datama = np.zeros((data.shape[0], data.shape[1]+3))

  for i in range(len(data)):
    datama[i] = np.append(data[i], ma_dict[str(datetime.datetime(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])))])
  
  return datama


def rfr_test_predict(combined_data, combined_labels, test_data, n_est):
  rfr = RandomForestRegressor(n_estimators=n_est)
  rfr.fit(combined_data, combined_labels)
  pred_labels = rfr.predict(test_data)
  
  print 'Predicted test labels shape:', pred_labels.shape
  return pred_labels


def additional_passes(passes, test_labels_pass1, n_est, p = 1):
  passes -= 1
  p += 1
  # Reinit
  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = data_init()


  ## find ma of last 2 hours and add to data
  test_set = np.zeros((test_data_rf.shape[0], 7))
  for i in range(len(test_data_rf)):
    test_set[i] = np.append(test_data_rf[i,:4], test_labels_pass1[i])


  train_set = np.zeros((train_data_rf.shape[0]+dev_data_rf.shape[0], 7))
  for i in range(len(np.vstack((train_data_rf, dev_data_rf)))):
    train_set[i] = np.append(np.vstack((train_data_rf, dev_data_rf))[i,:4], np.vstack((train_labels, dev_labels))[i])


  total_set = np.vstack((train_set, test_set))

  label_dict = {}
  ma_dict = {}
  for n in range(len(total_set)):
    label_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = total_set[n, 4:]

  for n in range(len(total_set)):
    ma = [0.0, 0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last 2 hours is missing, use zeros
      ma = [0.0, 0.0, 0.0]
    ma_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = ma


  train_data_pass2 = add_ma_to_data(train_data_rf, ma_dict)
  dev_data_pass2 = add_ma_to_data(dev_data_rf, ma_dict)
  test_data_pass2 = add_ma_to_data(test_data_rf, ma_dict)
  column_names_pass2 = column_names_rf + ['ma_cas', 'ma_reg', 'ma_count']
  ##

  #remove day of month
  train_data_pass2, dev_data_pass2, test_data_pass2, column_names_pass2 = remove_dom(train_data_pass2, dev_data_pass2, test_data_pass2, column_names_pass2)


  print 'Test data shape: ', test_data_pass2.shape

  # Execute RFR with predicted test_labels in the test_data set
  print_div_line('RFR - pass %d' % (p))
  test_log_labels_pass2 = rfr_test_predict(np.vstack((train_data_pass2, dev_data_pass2)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2, n_est)
  print


  count_pred = np.exp(test_log_labels_pass2) - 1
  print count_pred.shape
  
  if passes == 0:
    return count_pred
  else:
    return additional_passes(passes, count_pred, n_est, p)

def multiple_passes_rfr():
  ## multiple passes

  # In[ ]:

  n_est = 300

  ##pass 1
  #data init
  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = data_init()

  #remove day of month
  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = remove_dom(train_data_rf, dev_data_rf, test_data_rf, column_names_rf)

  # Execute RFR 
  print_div_line('RFR - pass 1')
  test_log_labels_pass1 = rfr_test_predict(np.vstack((train_data_rf, dev_data_rf)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_rf, n_est)
  test_labels_pass1 = np.exp(test_log_labels_pass1) - 1

  ##additional passes
  out_labels = additional_passes(4, test_labels_pass1, n_est)

  #prepare submission
  out_labels = out_labels[:,2]
  print "%d predicted count values <= 0" % len(out_labels[out_labels<0])
    
  for i in range(len(out_labels)):
    if out_labels[i] < 0:
      out_labels[i] = 0

  print out_labels.shape

  #write_submission(out_labels)


## gradient tree boosting

# The size of the regression tree base learners defines the level of variable interactions that can be captured by the gradient boosting model. In general, a tree of depth h can capture interactions of order h . If you specify max_depth=h then complete binary trees of depth h will be grown. Such trees will have (at most) 2**h leaf nodes and 2**h - 1 split nodes.
# 
# Alternatively, you can control the tree size by specifying the number of leaf nodes via the parameter max_leaf_nodes. In this case, trees will be grown using best-first search where nodes with the highest improvement in impurity will be expanded first. A tree with max_leaf_nodes=k has k - 1 split nodes and thus can model interactions of up to order max_leaf_nodes - 1 .
# 
# __We found that max_leaf_nodes=k gives comparable results to max_depth=k-1 but is significantly faster to train at the expense of a slightly higher training error.__
# 
# learning_rate is a hyper-parameter in the range (0.0, 1.0]. learning_rate strongly interacts with the parameter n_estimators, the number of weak learners to fit. Smaller values of learning_rate require larger numbers of weak learners to maintain a constant training error. Empirical evidence suggests that small values of learning_rate favor better test error. recommend to set the learning rate to a small constant (e.g. learning_rate <= 0.1) and choose n_estimators by early stopping. 
# 
# proposed stochastic gradient boosting, which combines gradient boosting with bootstrap averaging (bagging). At each iteration the base classifier is trained on a fraction subsample of the available training data. The subsample is drawn without replacement. A typical value of subsample is 0.5. shrinkage outperforms no-shrinkage. Subsampling with shrinkage can further increase the accuracy of the model. Subsampling without shrinkage, on the other hand, does poorly.
# 
# http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting

# on variable interaction:
# 
# example: Interaction between adding sugar to coffee and stirring the coffee. Neither of the two individual variables has much effect on sweetness but a combination of the two does.
#   
# Given n predictors, the number of terms in a linear model that includes a constant, every predictor, and every possible interaction is 2^n . Since this quantity grows exponentially, it readily becomes impractically large. One method to limit the size of the model is to limit the order of interactions. http://en.wikipedia.org/wiki/Interaction_%28statistics%29#Model_size
# 
# according to the table, if we assume max 3-way interaction, we can set max_leaf_nodes to 300 - 500 for ~13 predictors

#### note: none of the models have done any better with holiday and workingday combined. so i am removing it from the data prep flow.

# In[ ]:

def eval_function_GBR(traindata, trainlabel, testdata, testlabels, colnames=[], n_est = [500], print_features = False, print_plot = False, learning_rate=0.05, max_leaf_nodes=300, subsample=0.5):
  print "n\t\tc\tr"
  for n in n_est:
    GBR = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n, max_leaf_nodes=max_leaf_nodes, subsample=subsample)
    GBR.fit(traindata, trainlabel)

    GBR_log_pred_labels = GBR.predict(testdata)

    print "%8.3f\t%7.5f\t%7.5f" % (n, RMSLE(GBR_log_pred_labels, testlabels), RMSLE(GBR_log_pred_labels, testlabels))
    if n == n_est[-1] and print_features == True:
      print "\nfeature importances:"
      for name, feature in zip(colnames, GBR.feature_importances_):
        print "%10s\t%6.4f"%(name, feature)
  if print_plot:
    act_vs_pred_plot(np.exp(GBR_log_pred_labels)-1,np.exp(testlabels)-1)

def data_prep_for_gbr_and_eval():
  #restart
  n_est = [30]

  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape

  #remove day of month
  train_data_gb = remove_dayofmonth(train_data_gb)
  dev_data_gb = remove_dayofmonth(dev_data_gb)
  test_data_gb = remove_dayofmonth(test_data_gb)
  column_names_gb = column_names_gb[:2] + column_names_gb[3:]
  print train_data_gb.shape

  eval_function_GBR(train_data_gb, train_labels_t1[:,0], dev_data_gb, dev_labels_t1[:,0], column_names_gb, [500], True, False, 0.06)
  eval_function_GBR(train_data_gb, train_labels_t1[:,1], dev_data_gb, dev_labels_t1[:,1], column_names_gb, [700], True, False, 0.03)

def tune_gbr():
  # ###looking at 'registered'
  # 0.03 learning rate, 700 estimators 

  # In[ ]:

  l_rate = [0.03, 0.04, 0.05]
  n_est = [100, 500, 700, 800, 1000]
  for l in l_rate:
    print
    for n in n_est:
      GBR = GradientBoostingRegressor(learning_rate=l, n_estimators=n, max_leaf_nodes=300, subsample=0.5)
      GBR.fit(train_data_gb, train_labels_t1[:,1])
      log_pred_labels = GBR.predict(dev_data_gb)

      #print 'Predicted test labels shape:', log_pred_labels.shape
      
      print "%5.2f\t%5.1f\t\t%7.5f" % (l, n, RMSLE(log_pred_labels, dev_labels_t1[:,1]))


  # ###looking at 'casual'
  # 0.06 learning rate, 500 estimators

  # In[ ]:

  l_rate = [0.05, 0.06, 0.07]
  n_est = [300, 500, 700, 1000]
  for l in l_rate:
    print
    for n in n_est:
      GBR = GradientBoostingRegressor(learning_rate=l, n_estimators=n, max_leaf_nodes=300, subsample=0.5)
      GBR.fit(train_data_gb, train_labels_t1[:,0])
      log_pred_labels = GBR.predict(dev_data_gb)

      #print 'Predicted test labels shape:', log_pred_labels.shape
      
      print "%5.2f\t%5.1f\t\t%7.5f" % (l, n, RMSLE(log_pred_labels, dev_labels_t1[:,0]))

def gbr_with_no_ma():
  ### train model on train and dev, predict test, submit

  # In[ ]:

  #casual
  GBR = GradientBoostingRegressor(learning_rate=0.03, n_estimators=700, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(np.vstack((train_data_gb, dev_data_gb)), np.append(train_labels_t1[:,0], dev_labels_t1[:,0]))
  log_pred_labels_casual = GBR.predict(test_data_gb)

  #registered
  GBR = GradientBoostingRegressor(learning_rate=0.05, n_estimators=500, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(np.vstack((train_data_gb, dev_data_gb)), np.append(train_labels_t1[:,1], dev_labels_t1[:,1]))
  log_pred_labels_registered = GBR.predict(test_data_gb)

  pred_labels_count = (np.exp(log_pred_labels_casual) - 1) + (np.exp(log_pred_labels_registered) - 1)

  print pred_labels_count.shape

  #write_submission(pred_labels_count)

# In[ ]:

def add_ma_to_data(data, ma_dict):
  datama = np.zeros((data.shape[0], data.shape[1]+2))

  for i in range(len(data)):
    datama[i] = np.append(data[i], ma_dict[str(datetime.datetime(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])))])
  
  return datama
def gbr_test_predict(combined_data, combined_labels, test_data):
  #casual
  GBR = GradientBoostingRegressor(learning_rate=0.03, n_estimators=700, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,0])
  log_pred_labels_casual = GBR.predict(test_data)

  #registered
  GBR = GradientBoostingRegressor(learning_rate=0.06, n_estimators=500, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,1])
  log_pred_labels_registered = GBR.predict(test_data)

  log_pred_labels = np.vstack((log_pred_labels_casual, log_pred_labels_registered)).T
  
  print 'Predicted test labels shape:', log_pred_labels.shape
  return log_pred_labels


def gbr_double_pass():
  ### double pass with gbr

  ##prep data
  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape

  #remove day of month
  train_data_gb = remove_dayofmonth(train_data_gb)
  dev_data_gb = remove_dayofmonth(dev_data_gb)
  test_data_gb = remove_dayofmonth(test_data_gb)
  column_names_gb = column_names_gb[:2] + column_names_gb[3:]
  print train_data_gb.shape

  #add temp from last hour
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_temp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_temp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_temp)

  column_names_gb = column_names_gb + ['last_temp']
  print train_data_gb.shape

  #add atemp from last hour
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_atemp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_atemp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_atemp)

  column_names_gb = column_names_gb + ['last_atemp']
  print train_data_gb.shape


  # Execute GBR 
  print_div_line('GBR - pass 1')
  test_log_labels_pass1 = gbr_test_predict(np.vstack((train_data_gb, dev_data_gb)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_gb)
  print


  # Reinit
  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape

  #add temp from last hour
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_temp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_temp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_temp)

  column_names_gb = column_names_gb + ['last_temp']
  print train_data_gb.shape

  #add atemp from last hour
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_atemp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_atemp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_atemp)

  column_names_gb = column_names_gb + ['last_atemp']
  print train_data_gb.shape



  # Prepare data with moving average features
  test_labels_pass1 = np.exp(test_log_labels_pass1) - 1


  ## find ma of last 2 hours and add to data
  test_set = np.zeros((test_data_gb.shape[0], 6))
  for i in range(len(test_data_gb)):
    test_set[i] = np.append(test_data_gb[i,:4], test_labels_pass1[i])


  train_set = np.zeros((train_data_gb.shape[0]+dev_data_gb.shape[0], 6))
  for i in range(len(np.vstack((train_data_gb, dev_data_gb)))):
    train_set[i] = np.append(np.vstack((train_data_gb, dev_data_gb))[i,:4], np.vstack((train_labels[:,:2], dev_labels[:,:2]))[i])


  total_set = np.vstack((train_set, test_set))

  label_dict = {}
  ma_dict = {}
  for n in range(len(total_set)):
    label_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = total_set[n, 4:]

  for n in range(len(total_set)):
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last 2 hours is missing, use zeros
      ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = ma


  train_data_pass2 = add_ma_to_data(train_data_gb, ma_dict)
  dev_data_pass2 = add_ma_to_data(dev_data_gb, ma_dict)
  test_data_pass2 = add_ma_to_data(test_data_gb, ma_dict)
  ##



  #remove day of month
  train_data_pass2 = remove_dayofmonth(train_data_pass2)
  dev_data_pass2 = remove_dayofmonth(dev_data_pass2)
  test_data_pass2 = remove_dayofmonth(test_data_pass2)
  column_names_pass2 = np.append(column_names_gb[:2], column_names_gb[3:])
  print train_data_pass2.shape

  print 'Test data shape: ', test_data_pass2.shape

  # Execute RFR with predicted test_labels in the test_data set
  print_div_line('RFR - pass2')
  test_log_labels_pass2 = gbr_test_predict(np.vstack((train_data_pass2, dev_data_pass2)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2)
  print

  # Prepare submission
  cas_pred = np.exp(test_log_labels_pass2[:,0]) - 1
  reg_pred = np.exp(test_log_labels_pass2[:,1]) - 1
  count_pred = cas_pred + reg_pred

  print "%d predicted count values <= 0" % len(count_pred[count_pred<0])

  for i in range(len(count_pred)):
    if count_pred[i] < 0:
      count_pred[i] = 0

  print count_pred.shape
  #write_submission(count_pred)


## multi (5) pass on gbr

# In[ ]:

def data_init():
  
  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape

  #add temp from last hour
  yester_dict_temp = build_yester_dict_features(1, 8, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_temp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_temp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_temp)

  column_names_gb = column_names_gb + ['last_temp']
  print train_data_gb.shape

  #add atemp from last hour
  yester_dict_atemp = build_yester_dict_features(1, 9, ydict, np.vstack((train_data, dev_data, test_data)))

  train_data_gb = add_yesterhour_features(train_data_gb, yester_dict_atemp)
  dev_data_gb = add_yesterhour_features(dev_data_gb, yester_dict_atemp)
  test_data_gb = add_yesterhour_features(test_data_gb, yester_dict_atemp)

  column_names_gb = column_names_gb + ['last_atemp']
  print train_data_gb.shape
  
  return train_data_gb, dev_data_gb, test_data_gb, column_names_gb


def remove_dom(train_data_rf, dev_data_rf, test_data_rf, column_names_rf):

  #remove day of month
  train_data_rf = remove_dayofmonth(train_data_rf)
  dev_data_rf = remove_dayofmonth(dev_data_rf)
  test_data_rf = remove_dayofmonth(test_data_rf)
  column_names_rf = column_names_rf[:2] + column_names_rf[3:]
  print train_data_rf.shape
  
  return train_data_rf, dev_data_rf, test_data_rf, column_names_rf


def add_ma_to_data(data, ma_dict):
  datama = np.zeros((data.shape[0], data.shape[1]+2))

  for i in range(len(data)):
    datama[i] = np.append(data[i], ma_dict[str(datetime.datetime(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])))])
  
  return datama


def gbr_test_predict(combined_data, combined_labels, test_data):
  #casual
  GBR = GradientBoostingRegressor(learning_rate=0.03, n_estimators=700, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,0])
  log_pred_labels_casual = GBR.predict(test_data)

  #registered
  GBR = GradientBoostingRegressor(learning_rate=0.06, n_estimators=500, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,1])
  log_pred_labels_registered = GBR.predict(test_data)

  log_pred_labels = np.vstack((log_pred_labels_casual, log_pred_labels_registered)).T
  
  print 'Predicted test labels shape:', log_pred_labels.shape
  return log_pred_labels


def additional_passes(passes, test_labels_pass1, p = 1):
  passes -= 1
  p += 1
  # Reinit
  train_data_rf, dev_data_rf, test_data_rf, column_names_rf = data_init()


  ## find ma of last 2 hours and add to data
  test_set = np.zeros((test_data_rf.shape[0], 6))
  for i in range(len(test_data_rf)):
    test_set[i] = np.append(test_data_rf[i,:4], test_labels_pass1[i])


  train_set = np.zeros((train_data_rf.shape[0]+dev_data_rf.shape[0], 6))
  for i in range(len(np.vstack((train_data_rf, dev_data_rf)))):
    train_set[i] = np.append(np.vstack((train_data_rf, dev_data_rf))[i,:4], np.vstack((train_labels[:,:2], dev_labels[:,:2]))[i])


  total_set = np.vstack((train_set, test_set))

  label_dict = {}
  ma_dict = {}
  for n in range(len(total_set)):
    label_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = total_set[n, 4:]

  for n in range(len(total_set)):
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last 2 hours is missing, use zeros
      ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = ma


  train_data_pass2 = add_ma_to_data(train_data_rf, ma_dict)
  dev_data_pass2 = add_ma_to_data(dev_data_rf, ma_dict)
  test_data_pass2 = add_ma_to_data(test_data_rf, ma_dict)
  column_names_pass2 = column_names_rf + ['ma_cas', 'ma_reg']
  ##

  #remove day of month
  train_data_pass2, dev_data_pass2, test_data_pass2, column_names_pass2 = remove_dom(train_data_pass2, dev_data_pass2, test_data_pass2, column_names_pass2)


  print 'Test data shape: ', test_data_pass2.shape

  # Execute RFR with predicted test_labels in the test_data set
  print_div_line('GBR - pass %d' % (p))
  test_log_labels_pass2 = gbr_test_predict(np.vstack((train_data_pass2, dev_data_pass2)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2)
  print

  cas_pred = np.exp(test_log_labels_pass2[:,0]) - 1
  reg_pred = np.exp(test_log_labels_pass2[:,1]) - 1
  count_pred = np.vstack((cas_pred, reg_pred)).T

  print count_pred.shape
  
  if passes == 0:
    return count_pred
  else:
    return additional_passes(passes, count_pred, p)

def multi_pass_gbr():
  # In[ ]:

  ##pass 1
  #data init
  train_data_gb, dev_data_gb, test_data_gb, column_names_gb = data_init()

  #remove day of month
  train_data_gb, dev_data_gb, test_data_gb, column_names_gb = remove_dom(train_data_gb, dev_data_gb, test_data_gb, column_names_gb)

  # Execute GBR 
  print_div_line('GBR - pass 1')
  test_log_labels_pass1 = gbr_test_predict(np.vstack((train_data_gb, dev_data_gb)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_gb)
  test_labels_pass1 = np.exp(test_log_labels_pass1) - 1

  ##additional passes
  out_labels = additional_passes(4, test_labels_pass1)

  #prepare submission
  cas_labels = out_labels[:,0]
  reg_labels = out_labels[:,1]
  out_labels = cas_labels + reg_labels
  print "%d predicted count values <= 0" % len(out_labels[out_labels<0])
    
  for i in range(len(out_labels)):
    if out_labels[i] < 0:
      out_labels[i] = 0

  print out_labels.shape

  #write_submission(out_labels)


### double pass, with a before/after

#### also getting rid of last hour temp and atemp

# In[ ]:

def add_ma_to_data(data, ma_dict):
  datama = np.zeros((data.shape[0], data.shape[1]+2))

  for i in range(len(data)):
    datama[i] = np.append(data[i], ma_dict[str(datetime.datetime(int(data[i][0]), int(data[i][1]), int(data[i][2]), int(data[i][3])))])
  
  return datama
def gbr_test_predict(combined_data, combined_labels, test_data):
  #casual
  GBR = GradientBoostingRegressor(learning_rate=0.03, n_estimators=700, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,0])
  log_pred_labels_casual = GBR.predict(test_data)

  #registered
  GBR = GradientBoostingRegressor(learning_rate=0.06, n_estimators=500, max_leaf_nodes=300, subsample=0.5)
  GBR.fit(combined_data, combined_labels[:,1])
  log_pred_labels_registered = GBR.predict(test_data)

  log_pred_labels = np.vstack((log_pred_labels_casual, log_pred_labels_registered)).T
  
  print 'Predicted test labels shape:', log_pred_labels.shape
  return log_pred_labels

def double_pass_gbr_before_after():
  ##prep data
  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape

  #remove day of month
  train_data_gb = remove_dayofmonth(train_data_gb)
  dev_data_gb = remove_dayofmonth(dev_data_gb)
  test_data_gb = remove_dayofmonth(test_data_gb)
  column_names_gb = column_names_gb[:2] + column_names_gb[3:]
  print train_data_gb.shape


  # Execute GBR 
  print_div_line('GBR - pass 1')
  test_log_labels_pass1 = gbr_test_predict(np.vstack((train_data_gb, dev_data_gb)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_gb)
  print


  # Reinit
  #add day of week as a single attribute

  train_data_gb = add_dayofweek_as_single_attr(train_data)
  dev_data_gb = add_dayofweek_as_single_attr(dev_data)
  test_data_gb = add_dayofweek_as_single_attr(test_data)
  column_names_gb = column_names + ['weekday']
  print train_data_gb.shape


  # Prepare data with moving average features

  test_labels_pass1 = np.exp(test_log_labels_pass1) - 1


  ## find ma of last 2 hours and add to data
  test_set = np.zeros((test_data_gb.shape[0], 6))
  for i in range(len(test_data_gb)):
    test_set[i] = np.append(test_data_gb[i,:4], test_labels_pass1[i])


  train_set = np.zeros((train_data_gb.shape[0]+dev_data_gb.shape[0], 6))
  for i in range(len(np.vstack((train_data_gb, dev_data_gb)))):
    train_set[i] = np.append(np.vstack((train_data_gb, dev_data_gb))[i,:4], np.vstack((train_labels[:,:2], dev_labels[:,:2]))[i])


  total_set = np.vstack((train_set, test_set))

  label_dict = {}
  ma_dict = {}
  for n in range(len(total_set)):
    label_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = total_set[n, 4:]

  for n in range(len(total_set)):
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in range(2)]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from last 2 hours is missing, use zeros
      ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = ma


  train_data_pass2 = add_ma_to_data(train_data_gb, ma_dict)
  dev_data_pass2 = add_ma_to_data(dev_data_gb, ma_dict)
  test_data_pass2 = add_ma_to_data(test_data_gb, ma_dict)
  ##


  ## find ma of next 2 hours and add to data
  test_set = np.zeros((test_data_gb.shape[0], 6))
  for i in range(len(test_data_gb)):
    test_set[i] = np.append(test_data_gb[i,:4], test_labels_pass1[i])


  train_set = np.zeros((train_data_gb.shape[0]+dev_data_gb.shape[0], 6))
  for i in range(len(np.vstack((train_data_gb, dev_data_gb)))):
    train_set[i] = np.append(np.vstack((train_data_gb, dev_data_gb))[i,:4], np.vstack((train_labels[:,:2], dev_labels[:,:2]))[i])


  total_set = np.vstack((train_set, test_set))

  label_dict = {}
  ma_dict = {}
  for n in range(len(total_set)):
    label_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = total_set[n, 4:]

  for n in range(len(total_set)):
    ma = [0.0, 0.0]
    ma_date_list = [str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])) - datetime.timedelta(hours = h_i + 1)) for h_i in [-1,-2]]
    a = 0
    cnt = 0
    for ma_date in ma_date_list:
      if ma_date in label_dict:
        a += label_dict[ma_date]
        cnt += 1
    if cnt > 0:
      ma = (a/np.float(cnt))
    else:
      #if data from next 2 hours is missing, use zeros
      ma = [0.0, 0.0]
    ma_dict[str(datetime.datetime(int(total_set[n][0]), int(total_set[n][1]), int(total_set[n][2]), int(total_set[n][3])))] = ma


  train_data_pass2 = add_ma_to_data(train_data_pass2, ma_dict)
  dev_data_pass2 = add_ma_to_data(dev_data_pass2, ma_dict)
  test_data_pass2 = add_ma_to_data(test_data_pass2, ma_dict)
  ##


  #remove day of month
  train_data_pass2 = remove_dayofmonth(train_data_pass2)
  dev_data_pass2 = remove_dayofmonth(dev_data_pass2)
  test_data_pass2 = remove_dayofmonth(test_data_pass2)
  column_names_pass2 = np.append(column_names_gb[:2], column_names_gb[3:])
  print train_data_pass2.shape

  print 'Test data shape: ', test_data_pass2.shape

  # Execute GBR with predicted test_labels in the test_data set
  print_div_line('GBR - pass2')
  test_log_labels_pass2 = gbr_test_predict(np.vstack((train_data_pass2, dev_data_pass2)), np.vstack((train_labels_t1, dev_labels_t1)), test_data_pass2)
  print

  # Prepare submission
  cas_pred = np.exp(test_log_labels_pass2[:,0]) - 1
  reg_pred = np.exp(test_log_labels_pass2[:,1]) - 1
  count_pred = cas_pred + reg_pred

  print "%d predicted count values <= 0" % len(count_pred[count_pred<0])

  for i in range(len(count_pred)):
    if count_pred[i] < 0:
      count_pred[i] = 0

  print count_pred.shape
  #write_submission(count_pred)