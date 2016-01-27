__author__ = 'Trevor "Autogen" Grant'

"""
This is soooooo lazy.
"""
import pandas as pd

print "loading test data"
data = pd.DataFrame.from_csv("input_data/testing_soups.csv", index_col=None)

print "testing dates_one_hot"
from m6_local.functions import dates_one_hot
date_series = pd.to_datetime(data['DATE'])
one_hot_dates = dates_one_hot(date_series)
if one_hot_dates.shape == (100,16):
    print "looks good"
else:
    print "DOOD, this is broken!!!!"
