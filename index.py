import streamlit as st


import pickle
file1 = open("Linear_Regression.pkl",'rb') 
file2 = open("Standard_Scaler.pkl",'rb')
model = pickle.load(file1)  
ss = pickle.load(file2)  


def predict_price(features):    # predict_review is a user defined function
    # This step will convert the input msg into numbers
    val_ss = ss.transform([features])
    result = model.predict(val_ss)[0]
    return result

#----------------------------

# TEST
# This block of code is used to increase the font size of labels (area_income,Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population )

tabs_font_css = """
<style>
div[class*="stTextArea"] label p {
  font-size: 19px;
  color: red;
}

div[class*="stTextInput"] label p {
  font-size: 19px;
  color: blue;
}

div[class*="stNumberInput"] label p {
  font-size: 19px;
  color: green;
}
</style>
"""

st.write(tabs_font_css, unsafe_allow_html=True)


#-------------------------

st.write("# :green[House Price Prediction] ")
# message = st.text_area("Message")
area_income = st.number_input(label=':green[area_income]',step=1.,format="%.2f")

Avg_Area_House_Age = st.number_input(label=':green[Avg_Area_House_Age]',step=1.,format="%.2f")

Avg_Area_Number_of_Rooms = st.number_input(label=':green[Avg_Area_Number_of_Rooms]',step=1.,format="%.2f")

Avg_Area_Number_of_Bedrooms = st.number_input(label=':green[Avg_Area_Number_of_Bedrooms]',step=1.,format="%.2f")

Area_Population = st.number_input(label=':green[Area_Population]',step=1.,format="%.2f")

message = [area_income,Avg_Area_House_Age, Avg_Area_Number_of_Rooms, Avg_Area_Number_of_Bedrooms, Area_Population ]


if st.button(":green[Predict]"):
    result = predict_price(message)
    st.write(result)
    
    
    
 #================================================================
# NLP
    

file_1 = open("model_1.pkl",'rb') 
file_2 = open("CountVector.pkl",'rb')
model_1 = pickle.load(file_1)  
cv = pickle.load(file_2)  



def predict_review(msg):   
    msg_cv = cv.transform([msg])
    result_1 = model_1.predict(msg_cv)[0]
    return result_1



st.write("# :blue[Sentiment Analysis - NLP Project] ")
message = st.text_area(":blue[Message]")

if st.button(":blue[Predict_Sentiment]"):
    result_1 = predict_review(message)
    st.write(result_1)
    
    
    