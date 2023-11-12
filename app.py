import requests
import time
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import streamlit as st
from streamlit_lottie import st_lottie as stl
import pandas as pd
import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from model_infer import ModelInference

DATA_PATH = "dataset/data_daily.csv"
MODEL_PATH = "checkpoints/my_checkpoint"

st.set_page_config(page_title="Predictive Scanned Receipt Count", 
                   page_icon=":receipt:", layout="wide")

def lottie_load(url):
    r = requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)



local_css("style/style.css")

# Figures and animations
animation = lottie_load("https://lottie.host/839d8fa2-b863-4d77-84dd-5b1d9a9df1ed/g8T3P0Zs4T.json")
# training_data_img = Image.open("images/training_data.png")
# ModelResiduals_img = Image.open("images/modelExpectedVSPredicted.png")
# lossVSval_img = Image.open("images/lossVSval.png")
Improvement21vs22_img = Image.open("images/Improvement21vs22.png")
ExpectedVSPredicted_img = Image.open("images/ExpectedVSPredicted.png")
# box_hist_img = Image.open("images/box_hist.png")

# Introduction
with st.container():
    left_column, right_column = st.columns((2, 1))
    with left_column:
        st.header("Hi, I am Karen. Welcome to my website!! :blush:")
        st.subheader("This website contains my takeaway on the Fetch Rewards Take-home Exercise.")
        st.write("""I am a graduate research assistant pursuing a Master's in 
                     Computer Engineering at the University of Massachusetts Dartmouth (UMassD), 
                     from which I will graduate in December 2023, focusing on machine learning 
                     applications for predictive analytics and image processing (CNN and GAN) using Python and 
                     several frameworks such as TensorFlow, Keras, Scikit-Learn, NumPy, and Pandas. 
                     I have strong analytical and problem-solving skills, as well as quantitative research 
                     and predictive analytics experience. I am passionate about in designing, developing, 
                     and implementing new machine-learning applications to solve real-world problems and contribute 
                     to decision-making processes. Feel free to connect with me on my [LinkedIn](https://www.linkedin.com/in/karendamata/)!""")
    with right_column:
        stl(animation, height=300, key='animation')

# Instructions
with st.container():
    st.write("---")
    st.markdown("<h1 style='text-align: center; '>Problem Instructions</h1>", unsafe_allow_html=True)
    left_column, right_column = st.columns((1, 1))
    with left_column:
        st.write(
            """At fetch, the number of scanned receipts to their app on 
            a daily base is monitored as one of our KPIs. From a business standpoint, the possible
            predicted number of scanned receipts for a given future month is needed. 
            The number of observed scanned receipts each day for the year 2021 was provided. 
            Based on this prior knowledge, an algorithm that can predict the approximate number 
            of scanned receipts for each month of 2022 was developed.""")
    with right_column:
        st.write("""As mentioned in the left, 
                 The number of receipts scanned during the year of 2021 was provided. 
                 If you would like to run the trained model in a different data set, please upload 
                 a new csv.file containing two columns: # Date and Receipt_Count. 
                 # Date value must be formatted as YYYY-MM-DD.
                 Receipt_Count must be an integer number. See below an example of acceptable file.
                 """)
        file_example = """
        # Date,Receipt_Count
        2021-01-01,7564766
        2021-01-02,7455524
        2021-01-03,7095414
        2021-01-04,7666163"""
        st.code(file_example, language='txt')
        newdata_option = st.selectbox(
            'Would you like to upload a new data set?',
            ['No', 'Yes'], key='selection')

        if newdata_option=='Yes':
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                new_data = pd.read_csv(uploaded_file)
                if '# Date' and 'Receipt_Count' in list(new_data.columns):
                    new_data[['# Date', 'Receipt_Count']].to_csv("dataset/new.csv")
                    ModelInference(input_path="dataset/new.csv").model_results(training_size=0.8, training_required=False)
                    time.sleep(2)
                else:
                    st.write("###### Error: Incorrect Input! ######")

with st.container():
    st.write("---")
    st.write("""Before training a model, an exploratory data analysis was performed on the data provided to identify general patterns and identify
                data processing procedures needed. The data does not have missing value and appears to be normally distributed. Moreover, there is a 
                a very high and positive correlation between the dates and the number of receipts scanned. For this reason, At first look, a small
                 Neural Network was built, since the data is relatively small and has a strong correlation between the data provided. The data was 
                 then normalized and split into two parts for model validation. Since the intention of this website is to run inference on the model, 
                 the results of these steps are not shown here. If the reader is interested in seeing these results, please refer to the github repository.
                 The plot results are shown in the ["images" folder](https://github.com/Karendamata/ML-app-ReceiptCount/tree/main/images). """)
    st.write("---")
    st.markdown("<h1 style='text-align: center; '> Results</h1>", unsafe_allow_html=True)
    st.write("""
                The plot on the left shows the number of receipts scanned during the year 2021 in blue, and the model predictions are shown in red. 
                For many, the performance of the model considering the training data set is not important. However, important insight about the model can be 
                taken from this training data set. In this case, it seems that the model can learn the pattern in the data. The top plot on the right shows
                the monthly scanned receipt count during the previous year (2021) and the predicted count for the upcoming year (2022). The bottom plot on 
                the right shows the predicted increase (green) and decrease (red) in the monthly receipt count for the upcoming year. This plot specifically 
                is very useful for identifying possible changes in the company's demand, facilitating risk assessment, and planning 
                activities that might improve the company's business.
    """)
    left_column, right_column = st.columns(2)
    with left_column:
        st.image(ExpectedVSPredicted_img)
    with right_column:
        st.image(Improvement21vs22_img)

# Contact form
with st.container():
    st.write("---")
    st.write("Feel free to contact me!")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/damatakaren@yahoo.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
