from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model = load_model('Final_model')

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    from PIL import Image
    image = Image.open('logo.png')
    image_core = Image.open('logo.png')

    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to verify qualified customers')
    st.sidebar.success('https://www.pycaret.org')
    
    st.sidebar.image(image)

    st.title("Loan Verification  App")

    if add_selectbox == 'Online':
        MaritalStatus_B = st.selectbox('MaritalStatus_B', ['Married', 'Single','Divorced','Widow'])
        Gender_B = st.selectbox('Gender_B', ['M', 'F'])
        Location_B = st.selectbox('Location_B', ['Urban', 'SemiUrban', 'Rural'])
        EmployemtStatus_B = st.selectbox('EmployemtStatus_B', ['Unemployed', 'Worker', 'Employer','SelfEmployed'])
        Credit_score=st.number_input('Credit_score', min_value=1, max_value=1000, value=25)
        No_of_Dependents=st.number_input('No_of_Dependents', min_value=1, max_value=100, value=25)
        Age=st.number_input('Age', min_value=1, max_value=100, value=25)
        Available_balance=st.number_input('Available_balance', min_value=1, max_value=10000000000, value=25)
        Ledger_balance=st.number_input('Ledger_balance', min_value=1, max_value=100000000000000, value=25)
    
        
        output=""

        input_dict = {'MaritalStatus_B' : MaritalStatus_B, 'Gender_B' : Gender_B, 'Location_B':Location_B,'EmployemtStatus_B' :EmployemtStatus_B,'Credit_score': Credit_score, 'No_of_Dependents' : No_of_Dependents, 'Age' : Age , 'Available_balance':Available_balance, 'Ledger_balance' : Ledger_balance}
          
        
        input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model=model , input_df = input_df)
        output =  str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()