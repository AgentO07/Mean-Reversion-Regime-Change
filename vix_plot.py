import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# file VIX_History.csv 

df = pd.read_csv('VIX_History.csv', parse_dates=['DATE'])
df = df.sort_values('DATE').reset_index(drop=True) #not really needed since the csv is already sorted


# Plotting it

print(df.head())

plt.figure(figsize=(50,10)) # how many inches wide
plt.plot(df['DATE'], df['CLOSE'])
plt.title('Daily VIX Close Time Series')
plt.ylabel('Vix')
plt.xlabel('Date')
plt.show()