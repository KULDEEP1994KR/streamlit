
import streamlit as st
import pickle
import numpy as np
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

# Add more Streamlit components here as needed
def make_prediction(input_data):
    prediction = model.predict(np.array(input_data).reshape(-1, 1))
    return prediction


def predict1():
    st.markdown('<h1 style="color: white;">Linear Regression Prediction</h1>', unsafe_allow_html=True)
    st.write("Home Price Pridiction with area")
    
    # Input features for prediction
    feature = st.slider('Feature', min_value=0.0, max_value=1.0, step=0.01)
    
    # Example of using the model to make predictions
    input_data = [feature]  # Assuming your model requires only one feature
    prediction = make_prediction(input_data)
    
    st.write('Prediction:', prediction)
if __name__ == "__main__":
    main()



