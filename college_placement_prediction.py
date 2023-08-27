# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:16:22 2023

@author: Hp
"""

import numpy as np
import pickle as pk
import streamlit as st

loaded_model=pk.load(open('trained_model.sav','rb'))

def placement(input_data):
    input_data_array=np.asarray(input_data)
    input_data_array_new=np.reshape(input_data_array,(1,-1))
    predict=loaded_model.predict(input_data_array_new)
    if predict==0:
        return 'No Student Will Not Get College Placement'
    else:
        return 'Yes Student Will Get College Placement'
        
def main():
    st.title('Student College Placement Prediction By Machine Learning')
    CGPA=st.number_input('Enter CGPA Of Student Between 0 To 10')
    IQ=st.number_input('Enter IQ Of Student Between 0 To 130 And Above')
    prediction_placement=' '
    if st.button('Click Here To Check Whether Student Will Get College Placement Or Not'):
        prediction_placement=placement([CGPA,IQ])
    st.success(prediction_placement)
    st.subheader('Exploratory Data Analysis Done And Machine Learning Model Deployed By "Anubhav Kumar Gupta"')

if __name__=='__main__':
    main()
