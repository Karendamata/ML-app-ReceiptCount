
# ML-App to Predict Scanned Receipt Counts

This project implements a Machine Learning Application to predict the possible number of receipts that will be scanned in the upcoming year.






## Data
The data set used for model training contains the daily number of receipts scanned during the year 2021 (Receipt_Count) and the dates (# Date). An **exploratory data analysis** was performed to identify preprocessing procedures needed.

For **Preprocessing**, the data was normalized and split into two parts for training.
## Model
A neural network was implemented using TensorFlow to predict the Receip_Count values for the year 2022. Since the Receipt_Count data is relatively small and highly correlated to the # Dates, a small architecture shown bellow was chosen to avoid overfitting.


    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    dense (Dense)               (None, 3)                 12        
                                                                    
    dense_1 (Dense)             (None, 6)                 24        
                                                                    
    dense_2 (Dense)             (None, 3)                 21        
                                                                    
    dense_3 (Dense)             (None, 1)                 4         
                                                                    
    =================================================================
    Total params: 61 (244.00 Byte)
    Trainable params: 61 (244.00 Byte)
    Non-trainable params: 0 (0.00 Byte)

The model was trained for several epochs with an earlier stopping condition when the change in the loss did not improve by more than 0.0001 for twenty epochs. Then, the model performance was analyzed and validated.


## Web Application - How to run it?

The web application to run inferences on the trained model and to visualized the results was implemented with Streamlit and is stored in a Docker container. To run it, Docker must be installed, then run in your Command Prompt the following command: 

* docker compose up --build

Make sure to be in the same directory as the project. 

** ***The Application will be accessible on http://localhost:8501/*** **

More instructions are written in the web application. The results initially shown were achieved using the 2021 data set. A new data set can be uploaded for more results. Here is what it will look like: 
![Display Image](https://github.com/Karendamata/ML-app-ReceiptCount/tree/main/images/website.png)
## Authors

- [@karendamata](https://www.github.com/karendamata)

