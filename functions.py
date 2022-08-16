import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from collections import deque

def classify(current, future):
    if float(future) >float(current):
        return 1
    else:
        return 0

def shift_target(df, stock, seq_len, future_pred):
    df = df[[f"Close_{stock}", f"Volume_{stock}"]].copy()
    df['future'] = df[f"Close_{stock}"].shift(-future_pred) # Shifting the Price Variable, (From the most recent) 
    df['target'] = list(map(classify,df[f"Close_{stock}"],df['future'])) # Classifying Current Price with Future Price (Lagged Vs. Current)
    return df

def prep_df(df, seq_len):
    df = df.drop('future',axis=1)
    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change() # gets the percent change 
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values) # Scaling the data. 
    df.dropna(inplace=True)

    seq_data = []
    prev_days = deque(maxlen = seq_len) # appends all the data to the empty list, as it gets new items it deletes the old items 

    for i in df.values:
        prev_days.append([n for n in i[:-1]]) # each of the columns up to the last column (target)
        if len(prev_days) == seq_len:
            seq_data.append([np.array(prev_days), i[-1]])
    np.random.shuffle(seq_data)

    # balance the data 
    buys = []
    sells = []
    for seq, target in seq_data:
        if target== 0:
            sells.append([seq,target])
        elif target ==1:
            buys.append([seq, target])

    np.random.shuffle(buys)
    np.random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    seq_data = buys+sells
    np.random.shuffle(seq_data)

    X = []
    y = []
    
    for seq, target in seq_data:
        X.append(seq)
        y.append(target)
    X, y = np.array(X),np.array(y)
    print('Train Shape: ', X.shape, 'Target Shape: ', y.shape)
    return(X,y)