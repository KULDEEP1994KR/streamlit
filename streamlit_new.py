
import streamlit as st
import pickle
import numpy as np
from  langchain_openai import ChatOpenAI
import os

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


def main():
   
    
    image_css="""
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://wallpapercave.com/wp/wp3205272.jpg");
        background-size: cover;
    }
    [data-testid="stSidebar"]{
     background-image: url("https://img.freepik.com/free-photo/ai-technology-brain-background-digital-transformation-concept_53876-125206.jpg");
        background-size: cover;
    }
    
    }
    </style>
    """

    st.markdown(image_css, unsafe_allow_html=True)
    # st.title("THROUGH AI WE CAN PREDICT AND FORECAST ANYTHING")
    st.markdown('<h1 style="color: white;">THROUGH AI WE CAN PREDICT AND FORECAST ANYTHING</h1>', unsafe_allow_html=True)
    st.write("AI has indeed made significant strides in predictive analytics across various fields, from weather forecasting and financial markets to healthcare and customer behavior. With access to vast amounts of data and advanced algorithms, AI can often identify patterns and trends that humans might overlook, leading to more accurate predictions in many cases. However, it's essential to remember that predictions are based on probabilities and can never guarantee certainty. Additionally, ethical considerations and potential biases in data must be carefully addressed to ensure responsible and fair use of predictive AI technologies.")
    st.sidebar.markdown('<h1 style="font-family: sans-serif; color: white;">COMING SOON</h1>', unsafe_allow_html=True)
    predict1()



os.environ["OPENAI_API_KEY"]= "sk-proj-0uY2W00u9BkZK2IIEVFVT3BlbkFJHC22MEogyR"
llm= ChatOpenAI(temperature=0.7)


# Display the text input with custom CSS styling


st.markdown('<span style="font-family: Arial; font-size: 35px; color: white; font-weight: bold;">How may i help you:</span>', unsafe_allow_html=True)
llm_string = st.text_input(" ")
button_click=st.button("Ask GPT")


if button_click:
    if llm_string is not None:
         result= llm.invoke(llm_string)

         st.write("Results:")
         st.write(result)
        # st.info(result)
    else:
        st.warning("Enter some text to genrate results")




# Add more Streamlit components here as needed
def make_prediction(input_data):
    prediction = model.predict(np.array(input_data).reshape(-1, 1))
    return prediction


def predict1():
    st.markdown('<h1 style="color: white;">Linear Regression Prediction</h1>', unsafe_allow_html=True)
    # Input features for prediction
    # feature = st.slider('Feature', min_value=0.0, max_value=1.0, step=0.01)
    st.markdown('<span style="font-family: Arial; font-size: 25px;">Enter the area size to predict:</span>', unsafe_allow_html=True)

    feature = st.number_input(" ", value=0, step=1)
    
    # Example of using the model to make predictions
    input_data = [feature]  # Assuming your model requires only one feature
    prediction = make_prediction(input_data)
    
    st.write('Prediction:', prediction)
if __name__ == "__main__":
    main()



