import pandas as pd
import numpy as np
from datetime import timedelta
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from matplotlib import pyplot as plt
import matplotlib.dates as dates
import calendar

FIG_SIZE = (12,10)

class Processing():
    def __init__(self, df, transform=None):
        self.data = df
        self.m, self.n = df.shape
        self.output_variable = 'Receipt_Count'
        self.features = list('year', 'month', 'day')       

    def data_read(data_path):
        df_daily = pd.read_csv(data_path, parse_dates=True) # reading data from csv file
        list(df_daily.columns)
        df_daily[['year', 'month', 'day']] = df_daily['# Date'].str.split('-', expand=True)
        df_daily[['year', 'month', 'day']] = df_daily[['year', 'month', 'day']].apply(np.float64)
        df_daily['# Date'] = pd.to_datetime(df_daily['# Date'])
        df_daily.head()
        return df_daily
    
    def data_nxtYR(historical_df):
        df_data_nxtYR = pd.DataFrame()
        base = max(historical_df['# Date']) + timedelta(days=1)
        df_data_nxtYR['# Date'] = [base + timedelta(x) for x in range(365)]
        df_data_nxtYR['year'] = historical_df['year']+1
        df_data_nxtYR[['month', 'day']] = historical_df[['month', 'day']]
        df_data_nxtYR.head()
        return df_data_nxtYR

    # For testing purposes
    def twoyr_df(self, historical_df, nextyr_df):
        df_entire = pd.concat([historical_df, nextyr_df])
        df_entire = df_entire.drop(['Receipt_Count'], axis=1)
        return df_entire

    # To split data
    def data_split(self, historical_df, training_perc):
        df_training = historical_df.sample(frac=training_perc, random_state=0)
        df_testing = historical_df.drop(df_training.index)
        return df_training, df_testing
    
    # Function to normalize data
    # since there is no negative number in this data, dividing by the maximum value is enough for normalization
    def normalize(dataframe):
        return dataframe/dataframe.max(), dataframe.max()

    # Function to reverse normalization
    def inverse_normalize(dataframe, Max):
        return dataframe*Max
    

    def ead_plot(self, dataframe):
        plt.style.use('bmh')

        fig, ax  = plt.subplots(1, figsize=FIG_SIZE)
        ax.scatter(dataframe['# Date'], dataframe[self.output_variable])
        ax.set_title("Number of Daily Receipt Scanned in " + str(int(dataframe['year'][0])))
        plt.savefig("images/training_data.png")

        plt.style.use('bmh')

        fig, ax  = plt.subplots(1, 2, sharex=False, sharey=False, figsize=FIG_SIZE)
        ax[0].boxplot(dataframe[self.output_variable])
        ax[0].set_title('Box Plot of Receipt Counts')

        ax[1].hist(dataframe[self.output_variable])
        ax[1].set_title('Histogram of Receipt Counts')
        plt.savefig("images/box_hist.png")