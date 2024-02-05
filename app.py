# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 12:13:25 2023

@author: navaneethan
"""
# Load the saved KNN model
#model_filename = 'C:/Users/navaneethan/Desktop/mini project/Parkinson disease Prediction/saved models/parkinsons_knn_model.pkl'
#loaded_model = pickle.load(open(model_filename, 'rb'))

# Load the saved MinMaxScaler
#scaler_filename = 'C:/Users/navaneethan/Desktop/mini project/Parkinson disease Prediction/saved models/minmax_scaler.pkl'
#loaded_scaler = pickle.load(open(scaler_filename, 'rb'))


# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the saved KNN model and MinMaxScaler
model_filename = 'C:/Users/navaneethan/Desktop/mini project/Parkinson disease Prediction/saved models/parkinsons_knn_model.pkl'
loaded_model = pickle.load(open(model_filename, 'rb'))
scaler_filename = 'C:/Users/navaneethan/Desktop/mini project/Parkinson disease Prediction/saved models/minmax_scaler.pkl'
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

st.title('Parkinson\'s Disease Prediction')

# Define feature names
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE',
    'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]


# Apply CSS to style the "Clear Input Fields" button
st.markdown(
    "<style>"
    ".streamlit-button {"
    "    background-color: #f44336;"
    "    color: white;"
    "    padding: 10px 20px;"
    "    /* Add more CSS properties as needed */"
    "}"
    "</style>",
    unsafe_allow_html=True,
)

# Clear the input field values

# Create five columns for input fields
col1, col2, col3, col4, col5 = st.columns(5)

# Create input fields in the respective columns and collect feature values
features = []
for i, feature_name in enumerate(feature_names):
    if i % 5 == 0:
        with col1:
            feature_value = st.text_input(feature_name, key=feature_name, value=st.session_state.get(feature_name, ''))
    elif i % 5 == 1:
        with col2:
            feature_value = st.text_input(feature_name, key=feature_name, value=st.session_state.get(feature_name, ''))
    elif i % 5 == 2:
        with col3:
            feature_value = st.text_input(feature_name, key=feature_name, value=st.session_state.get(feature_name, ''))
    elif i % 5 == 3:
        with col4:
            feature_value = st.text_input(feature_name, key=feature_name, value=st.session_state.get(feature_name, ''))
    elif i % 5 == 4:
        with col5:
            feature_value = st.text_input(feature_name, key=feature_name, value=st.session_state.get(feature_name, ''))
    features.append(feature_value)

# Create a predict button
if st.button('Predict'):
    # Convert feature values to float
    if any(value == '' for value in features):
        st.warning('Please enter values for all input fields')
    else:
        # Convert feature values to float
        features = [float(value) for value in features]

        # Prepare the input data as a numpy array
        input_data = np.array(features).reshape(1, -1)

        # Scale the input data using the loaded scaler
        scaled_input_data = loaded_scaler.transform(input_data)

        # Make a prediction using the loaded KNN model
        prediction = loaded_model.predict(scaled_input_data)

        # Display the prediction result
        if prediction[0] == 0:
            st.write('Prediction: The person does not have Parkinson\'s Disease')
        else:
            st.write('Prediction: The person has Parkinson\'s Disease')
















