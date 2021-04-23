#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from textblob import TextBlob
from sklearn.decomposition import PCA
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('stock_price_xg.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    top=request.form.get("Top")
    openv=float(request.form.get("Open"))
    close=float(request.form.get("Close"))
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    feature = {'Top':top,
            'Open': openv,
            'Close': close }
    data = pd.DataFrame(feature, index=[0])
    data["polarity"] = data["Top"].map(lambda a: TextBlob(a).sentiment[0])
    data["subjectivity"] = data[f"Top"].map(lambda a: TextBlob(a).sentiment[1])
    #data.drop(["Top"], axis='columns', inplace=True)

    data_to_pred1 = {'polarity': data["polarity"],
                'subjectivity' : data["subjectivity"],
                'Open': data['Open'],
                'Close': data['Close'] }
    data_to_pred=pd.DataFrame(data_to_pred1, index=[0])

# Apply PCA
    X_train2 = np.load('X_train2.npy')
    pca = PCA(n_components=3)
    pca.fit(X_train2)

    transformed = pca.transform(data_to_pred)
    pca_df = pd.DataFrame(transformed)
    
    # Reads in saved classification model
    load_clf = pickle.load(open('stock_price_xg.pkl', 'rb'))
    predictions=load_clf.predict(pca_df)
    return render_template('index.html', prediction_text='The prediction of stock prices:1 indicate increasing while 0 decreasing. The prediction of these input is  : {}'.format(predictions[0]))
    


    #output = round(prediction[0], 2)

 

if __name__ == "__main__":
    app.run(debug=True)

