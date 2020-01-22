import os

import pandas as pd
import numpy as np

import pickle
import datetime as dt
from flask import Flask, jsonify, render_template

app = Flask(__name__)



#################################################
# Routes
#################################################

# Home Route
@app.route("/")
def home():
    return render_template("index.html")


    
# Route to run ML models and return results
@app.route("/predict/<selectedDateFormatted>")
def model(showDateFormatted):

    
    # Reformat date, extract weekday name and month number
    selectedDate = pd.to_datetime(selectedDateFormatted)
    selectedDateDT = pd.DatetimeIndex([selectedDate])
    selectedDay = selectedDateDT.day_name()[0] # Name of weekday
    selectedMonth = selectedDateDT.month[0] # Number of month

    # Determine season from month
    if (selectedMonth < 3 and selectedMonth > 11):
        selectedSeason = 4
    elif (selectedMonth < 9 and selectedMonth > 5):
        selectedSeason = 2
    elif (selectedMonth > 2 and selectedMonth < 6):
        selectedSeason = 1
    else:
        selectedSeason = 3

    # Determine working day from week
    if ( selectedDay == "Saturday" or selectedDay == "Sunday"):
        workingday = 0
    else:
         workingday = 1

    weekday=2
    holiday =0
    hr = 4
    weathersit=1
    temp=0.1
    hum=0.2
    windspeed=0.0896
    
    # Loading the saved model pickle
    model_pkl = open('models/RandomForestRegressor.pkl', 'rb')
    model = pickle.load(model_pkl)

    X_sample = np.asarray((season,holiday,selectedMonth, hr,weekday,workingday,weathersit,temp,	hum, windspeed)).astype(float)
    X_sample = X_sample.reshape(1,-1)

    prediction = model.predict(X_sample)

    # bike count prediction
    predicted_bike_count = prediction
  
    return predicted_bike_count

    

if __name__ == "__main__":
    app.run(debug=True)


