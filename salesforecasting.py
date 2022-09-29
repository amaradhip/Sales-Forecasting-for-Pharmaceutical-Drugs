import mysql.connector
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
from feature_engine.outliers import Winsorizer
from autots import AutoTS
from sklearn.pipeline import Pipeline
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose


############################################# MYSQL ######################################################

mydb = mysql.connector.connect(host="localhost", user="root", password="Amar@243", database="pharma")
query = "SELECT datum,M01AB,M01AE,N02BA,N02BE,N05B,N05C,R03,R06 FROM pharma.salesdaily;"

pharma = pd.read_sql(query, mydb)


######################################### AUTO EDA #############################################

profile = ProfileReport(pharma)
profile.to_file("pharma.html") 

# Box plot for finding outliers
sns.boxplot(pharma.M01AB)
sns.boxplot(pharma.M01AE)
sns.boxplot(pharma.N02BA)
sns.boxplot(pharma.N02BE)
sns.boxplot(pharma.N05B)
sns.boxplot(pharma.N05C)
sns.boxplot(pharma.R03)
sns.boxplot(pharma.R06)   
      


########################################## PRE-PROCESSING #################################################

def preprocessing():
    #Convert to datetime
    pharma["datum"] = pd.to_datetime(pharma["datum"])
    
    # columns
    columns = pharma.iloc[:,1:]
    
    # IQR
    IQR = pharma.quantile(0.75) - pharma.quantile(0.25)
    lower_limit = pharma.quantile(0.25) - (IQR * 1.5)
    upper_limit = pharma.quantile(0.75) + (IQR * 1.5)

    for i in columns:
        winsor = Winsorizer(capping_method='iqr',
                            tail='both',
                            fold=1.5,
                            variables=[i])
        pharma[i] = winsor.fit_transform(pharma[[i]])
        
        
    x = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']
    for i in x:
        result = seasonal_decompose(pharma[x].rolling(7, center=True).mean().dropna(), freq=7,model = "additive", filt=None)
        return pharma
        

################################ AUTO ML ####################################################

def model_def(drug_name):
    
    model_M01AB = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_M01AB = model_M01AB.fit(pharma,
        date_col = 'datum',
        value_col = 'M01AB',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_M01AB, open('model_M01AB.pkl', 'wb'))
    
    ## loading model to compare the results
    model_M01AB = pickle.load(open('model_M01AB.pkl', 'rb'))
    
 #################################################################################################   
    
    model_M01AE = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_M01AE = model_M01AE.fit(pharma,
        date_col = 'datum',
        value_col = 'M01AE',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_M01AE, open('model_M01AE.pkl', 'wb'))
    
    ## loading model to compare the results
    model_M01AE = pickle.load(open('model_M01AE.pkl', 'rb'))
    
####################################################################################################    
    
    model_N02BA = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_N02BA = model_N02BA.fit(pharma,
        date_col = 'datum',
        value_col = 'N02BA',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_N02BA, open('model_N02BA.pkl', 'wb'))
    
    ## loading model to compare the results
    model_N02BA = pickle.load(open('model_N02BA.pkl', 'rb'))
    
####################################################################################################    
    
    model_N02BE = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_N02BE = model_N02BE.fit(pharma,
        date_col = 'datum',
        value_col = 'N02BE',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_N02BE, open('model_N02BE.pkl', 'wb'))
    
    ## loading model to compare the results
    model_N02BE = pickle.load(open('model_N02BE.pkl', 'rb'))
     
    
###################################################################################################    
    
    
    model_N05B = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_N05B = model_N05B.fit(pharma,
        date_col = 'datum',
        value_col = 'N05B',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_N05B, open('model_N05B.pkl', 'wb'))
    
    ## loading model to compare the results
    model_N05B = pickle.load(open('model_N05B.pkl', 'rb'))
    
    
    
 ##################################################################################################   
    
    model_N05C = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_N05C = model_N05C.fit(pharma,
        date_col = 'datum',
        value_col = 'N05C',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_N05C, open('model_N05C.pkl', 'wb'))
    
    ## loading model to compare the results
    model_N05C = pickle.load(open('model_N05C.pkl', 'rb'))
    
    
#####################################################################################################    
    
    
    model_R03 = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_R03 = model_R03.fit(pharma,
        date_col = 'datum',
        value_col = 'R03',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_R03, open('model_R03.pkl', 'wb'))
    
    ## loading model to compare the results
    model_R03 = pickle.load(open('model_R03.pkl', 'rb'))
    
###################################################################################################    
    
    model_R06 = AutoTS(forecast_length = 7, frequency = 'infer',
        prediction_interval = 0.95,
        ensemble = None,
        model_list = "fast",  # "superfast", "default", "fast_parallel"
        transformer_list = "fast",  # "superfast",
        drop_most_recent = 1,
        max_generations = 4,
        num_validations = 2,
        validation_method = "backwards")
    model_R06 = model_R06.fit(pharma,
        date_col = 'datum',
        value_col = 'R06',
        id_col = None)
    
    ## saving model to disk
    pickle.dump(model_R06, open('model_R06.pkl', 'wb'))
    
    ## loading model to compare the results
    model_R06 = pickle.load(open('model_R06.pkl', 'rb'))
    
                                                                           
    
    return model_M01AB, model_M01AE, model_N02BA, model_N02BE, model_N05B, model_N05C, model_R03, model_R06
    
    
    

################################# Pipe line #################################################

pipe = Pipeline([('preprocessing', preprocessing(), 
                  'model_build', model_def("M01AB"), 
                  'model_build', model_def("M01AE"),
                  'model_build', model_def("N02BA"),
                  'model_build', model_def("N02BE"),
                  'model_build', model_def("N05B"),
                  'model_build', model_def("N05C"),
                  'model_build', model_def("R03"),
                  'model_build', model_def("R06"))])