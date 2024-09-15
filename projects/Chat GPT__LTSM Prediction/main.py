import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('aapl_stock_1yr.csv')
df.head()
df.tail(7)

df = df[['Date', 'Close']]
df.head()

#get rid of the $ symbol in our values
df = df.replace({'\$':''}, regex = True)

#convert 'Close' to a float, and 'Date' to a Date
df = df.astype({"Close": float})
df["Date"] = pd.to_datetime(df.Date, format="%m/%d/%Y")
df.dtypes
df.index = df['Date']
df.head()