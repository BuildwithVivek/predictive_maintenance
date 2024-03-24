import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from data_simulator import generate_data
from model import voting_clf, accuracy_dt, accuracy_nb, accuracy_lr, accuracy_vc



st.set_page_config(page_title="Predictive Maintenance", page_icon=":gear:", layout="wide")
page = st.sidebar.selectbox("Choose a page", ["Home","Failure Data", "Prediction", "Logs"])

df = pd.read_csv('predictive_maintenance.csv')
df.set_index('UDI', inplace=True)
df.drop(['Product ID', 'Target'], axis=1, inplace=True)

le = LabelEncoder()



if page == "Home":

    st.title("Predictive Maintenance : Machines")

    what_is_pm = """
    ### What is Predictive Maintenance?
    Predictive maintenance is a proactive approach employed by industries to anticipate and prevent potential equipment failures before they occur, thus minimizing downtime and optimizing operational efficiency. By leveraging advanced data analytics techniques, such as machine learning algorithms, predictive maintenance utilizes historical equipment data, real-time sensor readings, and other relevant factors to predict when equipment is likely to fail or require maintenance.
    """

    about_model = """
    ### About our model
    Our model is a **Voting Classifier**, made up of **Naive Bayes, Decision Tree, and Logistic Regression** classifiers,  trained on a dataset of machine failure data. It takes into account various features such as air temperature, process temperature, rotational speed, torque, and tool wear to predict the type of failure that may occur. The model has an accuracy of 0.966, which means it is able to predict the correct failure type 96.6% of the time. [Click here](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) to see the dataset used for training.
    """

    failure_types = """
    ### Types of Failures Predicted
    - Heat Dissipation Failure (Type 0)
    - No Failure (Type 1)
    - Overstrain Failure (Type 2)
    - Power Failure (Type 3)
    - Random Failures (Type 4)
    - Tool Wear Failure (Type 5)
    """

    # Display the content in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(what_is_pm)
    with col2:
        st.markdown(about_model)
    with col3:
        st.markdown(failure_types)

    st.markdown('### Model Performance')
    accuracy_df = {
    'Model': ['Decision Tree', 'Naive Bayes', 'Logistic Regression', 'Voting Classifier'],
    'Accuracy': [accuracy_dt * 100, accuracy_nb * 100, accuracy_lr * 100, accuracy_vc * 100]
        }

# Create a DataFrame
    df = pd.DataFrame(accuracy_df)

    # Display the DataFrame with reduced width
    st.write(df)

    st.markdown('### The Decision Tree')
    st.write("The decision tree below shows the rules used by our model to make predictions.")
    st.image('decision_tree_plot.png')

if page == "Failure Data":
    st.title("Failure Data")
    st.write("View the raw data at which the model was trained on, where the machine failed and the type of failure that occurred.")

    power_failure = df[df['Failure Type'] == 'Power Failure']
    heat_dissipation = df[df['Failure Type'] == 'Heat Dissipation Failure']
    overstrain = df[df['Failure Type'] == 'Overstrain Failure']
    random_failures = df[df['Failure Type'] == 'Random Failures']
    tool_wear = df[df['Failure Type'] == 'Tool Wear Failure']

    st.markdown("#### Power Failure")
    st.dataframe(power_failure, height=500)

    st.markdown("#### Heat Dissipation")
    st.dataframe(heat_dissipation, height=500)

    st.markdown("#### Overstrain")
    st.dataframe(overstrain, height=500)
    
    st.markdown("#### Random Failures")
    st.dataframe(random_failures, height=500)

    st.markdown("#### Tool Wear")
    st.dataframe(tool_wear, height=500)


if page == "Prediction":
#     st.markdown(
#     """
#     <style>
#     .css-1t42vg8 {
#         padding-top: 10px !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

    st.title("Use our model to make predictions!")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
                    The dataset consists of 10,000 data points stored as rows with the following features used in predictions:

                    1. **Type:** Consisting of a letter L(0), M(1), or H(2) for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number.
                    2. **air temperature [K]:** Generated using a random walk process later normalized to a standard deviation of 2 K around 300 K.
                    3. **process temperature [K]:** Generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K.
                    4. **rotational speed [rpm]:** Calculated from power of 2860 W, overlaid with a normally distributed noise.
                    5. **torque [Nm]:** Torque values are normally distributed around 40 Nm with an σ = 10 Nm and no negative values.
                    6. **tool wear [min]:** The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process.
                    7. **machine failure:** Label that indicates whether the machine has failed in this particular data point for any of the following failure modes are true.
                    """)

    def make_prediction(Type, air_temp, process_temp, rpm, torque, tool_wear):
        input_data = {
            'Type' : Type,
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rpm],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        }

        input_df = pd.DataFrame(input_data)
        prediction = voting_clf.predict(input_df)

        return prediction


    with col2:
        Type = st.selectbox('Type', ['0', '1', '2'])
        air_temp = st.slider('Air Temperature [K]', min_value=290, max_value=310, value=290)
        process_temp = st.slider('Process temperature [K]', min_value=300, max_value=320, value=300)
        rpm = st.slider('Rotational speed [rpm]', min_value=1000, max_value=2500, value=1000)
        torque = st.slider('Torque [Nm]', min_value=0, max_value=100, value=0)
        tool_wear = st.slider('Tool wear [min]', min_value=0, max_value=250, value=0)

    if st.button('Submit'):
        # Call the function to make predictions
        prediction = make_prediction(Type, air_temp, process_temp, rpm, torque, tool_wear)
        # Display the prediction
        st.text('Prediction: {}'.format(prediction))


if page == "Logs":
    st.title("Logs")

    columns_to_display = ['Product ID','Location','Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

   

    def display_logs():
        while True:

            random_data = generate_data()
            
            selected_data = random_data[columns_to_display]
            st.dataframe(selected_data, hide_index=True)

            random_data['Type'] = le.fit_transform(random_data[['Type']])
            input_row = random_data.drop(columns=['Failure Type', 'Product ID', 'Location'])

            prediction = voting_clf.predict(input_row)
            st.text('Prediction: {}'.format(prediction))
            
            # Wait for 2 seconds before displaying the next random data
            time.sleep(2)

    display_logs()
    # st.write("View logs here")