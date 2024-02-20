import streamlit as st
import pickle
import numpy as np

# import the model
rf = pickle.load(open("model/lreg_bbry_tuned_model.pkl",'rb'))


st.title("Yield Predictor")


clone_size = st.number_input("enter clone size: " , max_value=40.0 , key = "clone_size")

honeybee =  st.number_input("enter honeybee" , key = "honeybee")

bumbles =  st.number_input("enter bumbles" , key = "bumbles")

andrena =  st.number_input("enter andrena: ", key = "andrena")

osmia = st.number_input("enter osmia ", key = "osmia")

AverageRainingDays = st.number_input("enter AverageRainingDays: ", key = "raining_day")


submit = st.button('Predict Yield')
if submit:
    query = np.array([clone_size , honeybee , bumbles , andrena , osmia , AverageRainingDays])

    query = query.reshape(1,6)

    st.title("The predicted yield from randomforest is " + str(int((rf.predict(query)[0]))))