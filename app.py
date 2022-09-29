from flask import Flask,render_template,url_for,request, jsonify
import pandas as pd
import pickle
import joblib
import numpy as np


import os
os.chdir(r"C:\Users\91913\Documents\AMAR\Data Science\Project 1\Final Model")

app = Flask(__name__, template_folder = 'template')
model_M01AB = pickle.load(open('model_M01AB.pkl', 'rb'))
model_M01AE = pickle.load(open('model_M01AE.pkl', 'rb'))
model_N02BA = pickle.load(open('model_N02BA.pkl', 'rb'))
model_N02BE = pickle.load(open('model_N02BE.pkl', 'rb'))
model_N05B = pickle.load(open('model_N05B.pkl', 'rb'))
model_N05C = pickle.load(open('model_N05C.pkl', 'rb'))
model_R03 = pickle.load(open('model_R03.pkl', 'rb'))
model_R06 = pickle.load(open('model_R06.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
       
    #start_date = request.form.get('start_date')
    #end_date = request.form.get('end_date')
    
    
    #predict_date = pd.date_range(start = start_date, end = end_date)
    #pred_df = pd.DataFrame({'ds' : predict_date})
    
    prediction_M01AB = model_M01AB.predict()
    forecast_M01AB = prediction_M01AB.forecast
    forecast_M01AB = pd.DataFrame(forecast_M01AB).to_html()
    print(forecast_M01AB)
    
    prediction_M01AE = model_M01AE.predict()
    forecast_M01AE = prediction_M01AE.forecast
    forecast_M01AE = pd.DataFrame(forecast_M01AE).to_html()
    print(forecast_M01AE)
    
    prediction_N02BA = model_N02BA.predict()
    forecast_N02BA = prediction_N02BA.forecast
    forecast_N02BA = pd.DataFrame(forecast_N02BA).to_html()
    print(forecast_N02BA)
    
    prediction_N02BE = model_N02BE.predict()
    forecast_N02BE = prediction_N02BE.forecast
    forecast_N02BE = pd.DataFrame(forecast_N02BE).to_html()
    print(forecast_N02BE)
    
    prediction_N05B = model_N05B.predict()
    forecast_N05B = prediction_N05B.forecast
    forecast_N05B = pd.DataFrame(forecast_N05B).to_html()
    print(forecast_N05B)
    
    prediction_N05C = model_N05C.predict()
    forecast_N05C = prediction_N05C.forecast
    forecast_N05C = pd.DataFrame(forecast_N05C).to_html()
    print(forecast_N05C)
    
    prediction_R03 = model_R03.predict()
    forecast_R03 = prediction_R03.forecast
    forecast_R03 = pd.DataFrame(forecast_R03).to_html()
    print(forecast_R03)
    
    prediction_R06 = model_R06.predict()
    forecast_R06 = prediction_R06.forecast
    forecast_R06 = pd.DataFrame(forecast_R06).to_html()
    print(forecast_R06)
    
    text_file = open("index_result.html", "w")
    text_file.write(forecast_M01AB)
    text_file.write(forecast_M01AE)
    text_file.write(forecast_N02BA)
    text_file.write(forecast_N02BE)
    text_file.write(forecast_N05C)
    text_file.write(forecast_N05C)
    text_file.write(forecast_R03)
    text_file.write(forecast_R06)
    text_file.close()
    
    return render_template('index.html', prediction_text = (forecast_M01AB, forecast_M01AE,
                                       forecast_N02BA, forecast_N02BE,
                                       forecast_N05B, forecast_N05C,
                                       forecast_R03, forecast_R06))


if __name__ == "__main__":
    app.run(debug = True)
