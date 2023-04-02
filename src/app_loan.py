from flask import render_template, request, jsonify,Flask
import flask
import numpy as np
import traceback #allows you to send error to user
import pickle
import pandas as pd

# App definition
app = Flask(__name__)


# Define transformation
def encode_transform(df):
    #Convert object data type to numeric
    gender_stat = {"Female": 0, "Male": 1}
    yes_no_stat = {'No' : 0,'Yes' : 1}
    dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
    education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
    property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
    
    df['Gender']        = df['Gender'].replace(gender_stat)
    df['Married']       = df['Married'].replace(yes_no_stat)
    df['Dependents']    = df['Dependents'].replace(dependents_stat)
    df['Education']     = df['Education'].replace(education_stat)
    df['Self_Employed'] = df['Self_Employed'].replace(yes_no_stat)
    df['Property_Area'] = df['Property_Area'].replace(property_stat)
    
    # Convert object to numerics
    df['Dependents'] = pd.to_numeric(df['Dependents'])

    return df

# importing models
pkl_filename = 'loan_approval_model.pkl'
with open(pkl_filename, "rb" ) as f:
    trained_model = pickle.load (f)
    
#webpage
@app.route('/')
def welcome():
    return "Welcome to loan approval check app. Use 'POST' method with predict/params."

@app.route('/predict', methods=['POST','GET'])
def predict():

    if flask.request.method == 'GET':
       return "Prediction page. Use 'POST' method with predict/params."

    if flask.request.method == 'POST':
        try:
            json_data = request.json # '_' since 'json' is a special word
            df_data = encode_transform( pd.DataFrame(json_data) )
            prediction = list(trained_model.predict(df_data))[0]
                        
            message = "Loan IS NOT approved."
            if prediction=='Y': message = "Loan IS approved."
            
            return jsonify({  'Result ': message })

        except:
            return jsonify({ "ERROR: " : traceback.format_exc() })


if __name__ == "__main__":
    app.run('0.0.0.0', debug=True , port=5000)
