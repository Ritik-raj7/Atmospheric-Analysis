#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
from streamlit_jupyter import StreamlitPatcher
StreamlitPatcher().jupyter()  # Register Streamlit with Jupyter-compatible wrappers


# In[13]:



# -*- coding: utf-8 -*-
"""
Created on Fri May 03 03:25:36 2024

@author: Ritik Raj,Manoj Kumar,Manohar,Abhishek
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("random_forest_regression_model.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(T,TM,Tm,SLP,H,VV,V,VM):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: T
        in: query
        type: number
        required: true
      - name: TM
        in: query
        type: number
        required: true
      - name: Tm
        in: query
        type: number
        required: true
      - name: SLP
        in: query
        type: number
        required: true
      - name: H
        in: query
        type: number
        required: true
      - name: VV
        in: query
        type: number
        required: true
      - name: V
        in: query
        type: number
        required: true
      - name: VM
        in: query
        type: number
        required: true
      
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[T,TM,Tm,SLP,H,VV,V,VM]])
    print(prediction)
    return prediction



def main():
    st.title("Atmospheric Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Atmospheric Analysis ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    T = st.text_input("T","Average Temperature (°C)")
    TM = st.text_input("TM","Maximum temperature (°C)")
    Tm = st.text_input("Tm","Minimum temperature (°C)")
    SLP = st.text_input("SLP","Atmospheric pressure at sea level (hPa)")
    H = st.text_input("H","Average relative humidity (%)")
    VV = st.text_input("VV","Average visibility (Km)")
    V = st.text_input("V","	Average wind speed (Km/h)")
    VM = st.text_input("VM","Maximum sustained wind speed (Km/h)")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(T,TM,Tm,SLP,H,VV,V,VM)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.write("Air Quality Index (AQI) Values, Levels of Health Concern:")
        st.write("0 to 50 - Good")
        st.write("51 to 100 - Moderate")
        st.write("101 to 150 - Unhealthy for Sensitive Groups")
        st.write("151 to 200 - Unhealthy")
        st.write("201 to 300 - Very Unhealthy")
        st.write("301 to 500 - Hazardous")

if __name__=='__main__':
    main()
    
    
    


# In[ ]:




