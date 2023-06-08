import streamlit as st


import pickle
file_1 = open("model_1.pkl",'rb') 
file_2 = open("CountVector.pkl",'rb')
model_1 = pickle.load(file_1)  
cv = pickle.load(file_2)  

msg = 'It was amazing. I liked it a lot'

def predict_review(msg):   
    msg_cv = cv.transform([msg])
    result_1 = model_1.predict(msg_cv)[0]
    return result_1



st.write("# Hello World ")
message = st.text_area("Message")

if st.button("Predict"):
    result_1 = predict_review(message)
    st.write(result_1)