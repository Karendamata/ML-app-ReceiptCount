import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from matplotlib import pyplot as plt
import matplotlib.dates as dates
import calendar
from data_preprocessing import Processing
from model_def import MLP
# from model.utils import load_model


DATA_PATH = "dataset/data_daily.csv"
MODEL_PATH = "checkpoints/my_checkpoint"
FIG_SIZE = (12,10)

class ModelInference():
    def __init__(self, input_path):
        self.data = Processing.data_read(input_path)
        self.m, self.n = self.data.shape
        self.output_variable = 'Receipt_Count'
        self.features = list(self.data.columns)
        self.features.remove(self.output_variable)
        self.features.remove('# Date')
        self.features_num = len(self.features)
        self.training_required=False

    def model_results(self, training_size):
        print(self.data.head())
        print(self.data.shape)
        Processing.ead_plot(self, self.data)
        self.data_predictions = Processing.data_nxtYR(historical_df=self.data)
        print(self.data_predictions.head())
        print(self.data_predictions.shape)
        self.data_entire = Processing.twoyr_df(self, self.data, self.data_predictions)
        print(self.data_entire.head())
        print(self.data_entire.shape)

        self.df_training, self.df_testing = Processing.data_split(self, self.data, training_size)

        self.data_norm, self.data_max = Processing.normalize(self.data[self.features + [self.output_variable]])
        self.df_entire_normalized, self.df_entire_max = Processing.normalize(self.data_entire[self.features])
        self.df_train_norm, self.train_max = Processing.normalize(self.df_training[self.features + [self.output_variable]])
        self.df_test_norm, self.test_max = Processing.normalize(self.df_testing[self.features + [self.output_variable]])

        model = MLP()
        model.load_model(MODEL_PATH)
        if self.training_required!=False:
            training_history = model.training(self.df_train_norm, self.features, self.output_variable)
            
            # Analyzing training and validation losses to verify if overfitting has occurred.
            plt.style.use('bmh')
            fig, ax = plt.subplots(1, figsize=FIG_SIZE)
            ax.plot(training_history['loss'])
            ax.plot(training_history['val_loss'])
            ax.set_title("Training Loss x Validation Loss")
            plt.savefig("images/lossVSval_new.png")
            self.training_required==False

        pred_normalized = model.predict(self.df_entire_normalized, self.features)
        pred_not_normalized = Processing.inverse_normalize(pd.DataFrame(pred_normalized,columns=[self.output_variable]), self.data_max[[self.output_variable]])

        #Analyzing the model residuals
        plt.style.use('bmh')
        fig, ax  = plt.subplots(1, figsize=FIG_SIZE)
        ax.scatter(self.data[self.output_variable], pred_not_normalized[:365])
        line_coords = np.arange(self.data[self.output_variable].min().min(), self.data[self.output_variable].max().max())
        ax.plot(line_coords, line_coords, color='darkorange', linestyle='--')
        ax.set_title('Scatter Plot between the expected and predicted values.')
        plt.savefig("images/modelExpectedVSPredicted.png")

        self.data_entire.insert(1, self.output_variable , pred_not_normalized)

        # for business analysis
        #  summing up the number of receipts received at each month 
        monthly_2021 = self.data[['year', 'month',self.output_variable]].groupby(by='month').sum()[[self.output_variable]]
        monthly_2021.index = list(calendar.month_name)[1:]

        monthly_predicted = self.data_entire[365:][['year', 'month',self.output_variable]].groupby(by='month').sum()[[self.output_variable]]
        monthly_predicted.index = list(calendar.month_name)[1:]

        plt.style.use('bmh')
        fig, ax  = plt.subplots(1, figsize=FIG_SIZE)
        ax.plot(self.data['# Date'], self.data[[self.output_variable]], '.', label='Expected')
        ax.plot(self.data_entire['# Date'], self.data_entire[[self.output_variable]], label='Predicted')
        ax.xaxis.set_minor_locator(dates.MonthLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter('%b %Y'))
        ax.set_title('Expected vs Predicted Daily Receipt Counts')
        plt.legend()
        plt.savefig("images/ExpectedVSPredicted.png")

        # computing the difference of monthly receipt counts between 2021 and 2022
        diff = [value[0] for value in (monthly_predicted-monthly_2021).to_numpy()]

        # identifying the bar color for each month according to the difference of monthly receipt previous calculated
        # green represents an increase, and red a decrease
        colors = ['green' if diff[i]>0 else 'red' for i in range(len(diff))] 

        plt.style.use('bmh')
        fig, ax  = plt.subplots(2, sharex=True, sharey=False, figsize=FIG_SIZE)
        ax[0].plot(monthly_2021, label='2021')
        ax[0].plot(monthly_predicted, label=str(int(self.data_entire[365:]['year'][0])))
        ax[0].legend()
        ax[0].set_title('Monthly Receipt Count')
        ax[1].bar(monthly_predicted.index, diff, color=colors)
        ax[1].set_title('Predicted Difference on Receipt Count from 2021 to '+str(int(self.data_entire[365:]['year'][0])))
        plt.savefig("images/Improvement21vs22.png")
        
if __name__ == "__main__":
    ModelInference(input_path=DATA_PATH).model_results(training_size=0.8)