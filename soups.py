__author__ = 'Trevor "Autogen" Grant'

import numpy as np
import pandas as pd

from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.DataFrame.from_csv("C:/Users/tgrant/Documents/oos_pred_poc/soups.csv", index_col=None)
data = data[data['ACT_SALES_QTY'] > 0]
print "data loaded and filtered"
#data['DATE'] = pd.to_datetime(data['DATE'])
# need to handle dates teh honest way
data['fixed_dates'] = pd.to_datetime(data['DATE'])
print 'updated dates'
one_hot_store_ids = pd.get_dummies(data['STORE_ID'], sparse= True)
one_hot_item_ids = pd.get_dummies(data['ITEM_ID'], sparse=True)
one_hot_week_num = pd.get_dummies(data['fixed_dates'].apply(lambda x: x.strftime("%W")), sparse= True)
one_hot_weekday = pd.get_dummies(data['fixed_dates'].apply(lambda x: x.strftime("%w")), sparse= True)
one_hot_dom = pd.get_dummies(data['fixed_dates'].apply(lambda x: x.strftime("%d")), sparse= True)
one_hot_month = pd.get_dummies(data['fixed_dates'].apply(lambda x: x.strftime("%m")), sparse= True)
one_hot_year = pd.get_dummies(data['fixed_dates'].apply(lambda x: x.strftime("%Y")), sparse= True)
print 'created one-hots'
## These will come in handy for adding 'holiday' flags
# from pandas.tseries.holiday import USFederalHolidayCalendar, get_calendar
# get_calendar("USFederalHolidayCalendar")
#one_hot_dates = pd.get_dummies(data['DATE'], sparse=True)
# still need some other stuff - eg promo, act price/reg price
# need to normalize act_sales
# split training set
discount = data['PLAN_PRICE'] / data['REG_PRICE']

from sklearn.preprocessing import MinMaxScaler

from scipy.sparse import hstack
# need discount and data['ON_PROMOTION']
X = hstack([ one_hot_dom,one_hot_month, one_hot_week_num, one_hot_weekday, one_hot_year, one_hot_item_ids, one_hot_store_ids])


y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(np.array(data['ACT_SALES_QTY']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6)
from sklearn.linear_model import LinearRegression

clf = LinearRegression()
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
## Let's get some baseline scores.
for m in [explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error ]:
    print m.__name__, m(y, y_scaler.transform(data['FCAST_SALES_QTY'])), m(y_test, y_preds)

#not pure cheating
pd.DataFrame(y_scaler.inverse_transform(y_preds)).hist()