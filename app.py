# import tensorflowkeras
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
# import tensorflow_hub as hub
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from io import StringIO
# from tensorflow.keras import preprocessing

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="",
)
# loading the saved models

diabetes_model = pickle.load(open('C:/Users/rnegi/Desktop/ML complete/MLApp/savedmodels/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open('C:/Users/rnegi/Desktop/ML complete/MLApp/savedmodels/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open('C:/Users/rnegi/Desktop/ML complete/MLApp/savedmodels/parkinsons_model.sav', 'rb'))

# keras_model=load_model('C:/Users/rnegi/Desktop/ML complete/MLApp/nnmodels/tumour.h5')
# 2. horizontal menu
selected2 = option_menu(None, ["Home","Report"], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2=="Home":
    
# # sidebar for navigation
# st.title("Brain Tumour Detection App")
    with st.sidebar:
    
        selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Brain Tumor Prediction','Diabetes Prediction',
                           'Heart Disease Prediction'
                           ],
                          icons=['activity','heart','person'],
                          default_index=0)
    if selected == "Brain Tumor Prediction":
        try:
            st.subheader("Brain Tumor Prediction")
            image_file = st.file_uploader("Upload Image", type=['png', 'jpeg', 'jpg'])
            def predict(img):
                    model = load_model(r'C:/Users/rnegi/Desktop/ML complete/MLApp/nnmodels/tumour.h5',compile=False)
                    shape = ((200, 200, 3))
                    # model=tf.keras.Sequential([hub.kerasLayer(model, input_shape=shape)])
                    test_image = img.resize((200, 200))
                    image = np.reshape(test_image,[1,200,200,3])

                    tumour_prediction = model.predict(image)
                    return tumour_prediction
            
            if image_file is not None:
                # To See Details
                # st.write(type(image_file))
                # st.write(type(image_file))
                file_details = {"Filename": image_file.name, "FileType": image_file.type, "FileSize": image_file.size}
                st.write(file_details)
                global img
                img=Image.open(image_file)
                # img = load_image(image_file)
                tumour_prediction=predict(img)
                st.image(img)

            
            

                # code for Prediction
            tumour_diagnosis = ''

                # creating a button for Prediction

            if st.button('Tumour Test Result'):


                    if (tumour_prediction >= 0.5):
                        tumour_diagnosis = 'The person has a tumour'
                    else:
                        tumour_diagnosis = 'The person does not has a tumour'

            st.success(tumour_diagnosis)
        except:
            st.error("Please provide a valid input")
# Diabetes Prediction Page

    if selected == "Diabetes Prediction":
        try:
            # page title
            st.title('Diabetes Prediction using ML')
            
            
            # getting the input data from the user
            col1, col2, col3 = st.columns(3)
            
            with col1:
                Pregnancies = st.text_input('Number of Pregnancies')
                
            with col2:
                Glucose = st.text_input('Glucose Level')
            
            with col3:
                BloodPressure = st.text_input('Blood Pressure value')
            
            with col1:
                SkinThickness = st.text_input('Skin Thickness value')
            
            with col2:
                Insulin = st.text_input('Insulin Level')
            
            with col3:
                BMI = st.text_input('BMI value')
            
            with col1:
                DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
            
            with col2:
                Age = st.text_input('Age of the Person')

            
            # code for Prediction
            diab_diagnosis = ''
            
            # creating a button for Prediction
            import csv
            if st.button('Diabetes Test Result'):
                diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
                if (diab_prediction[0] == 1):
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                
                st.success(diab_diagnosis)
                a=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
                with open('my_file.csv', mode='a', newline='') as file:
                    for i in range(len(a)-2):
                        file.write(a[i]+',')
                    file.write(a[-1])
                    file.write('\n')
        except:
            st.error("Please provide a valid data")
            


    # Heart Disease Prediction Page
    if (selected == 'Heart Disease Prediction'):
        # try:
            # page title
            st.title('Heart Disease Prediction using ML')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.text_input('Age')
                
            with col2:
                sex = st.text_input('Sex')
                
            with col3:
                cp = st.text_input('Chest Pain types')
                
            with col1:
                trestbps = st.text_input('Resting Blood Pressure')
                
            with col2:
                chol = st.text_input('Serum Cholestoral in mg/dl')
                
            with col3:
                fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
                
            with col1:
                restecg = st.text_input('Resting Electrocardiographic results')
                
            with col2:
                thalach = st.text_input('Maximum Heart Rate achieved')
                
            with col3:
                exang = st.text_input('Exercise Induced Angina')
                
            with col1:
                oldpeak = st.text_input('ST depression induced by exercise')
                
            with col2:
                slope = st.text_input('Slope of the peak exercise ST segment')
                
            with col3:
                ca = st.text_input('Major vessels colored by flourosopy')
                
            with col1:
                thal = st.text_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect')
                
           
                        
            
            # code for Prediction
            heart_diagnosis = ''
            
            # creating a button for Prediction
            
            if st.button('Heart Disease Test Result'):
                age = int(age)
                sex = int(sex)
                cp = int(cp)
                trestbps = int(trestbps)
                chol = int(chol)
                fbs = int(fbs)
                restecg = int(restecg)
                thalach = int(thalach)
                exang = int(exang)
                oldpeak = float(oldpeak)
                slope = int(slope)
                ca = int(ca)
                thal = int(thal)
                heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
                
                if (heart_prediction[0] == 1):
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                
            st.success(heart_diagnosis)
        # except:
        #     st.error("Please provide a valid data")
        


    # Parkinson's Prediction Page
    # if (selected == "Parkinsons Prediction"):
        
    #     # page title
    #     st.title("Parkinson's Disease Prediction using ML")
        
    #     col1, col2, col3, col4, col5 = st.columns(5)  
        
    #     with col1:
    #         fo = st.text_input('MDVP:Fo(Hz)')
            
    #     with col2:
    #         fhi = st.text_input('MDVP:Fhi(Hz)')
            
    #     with col3:
    #         flo = st.text_input('MDVP:Flo(Hz)')
            
    #     with col4:
    #         Jitter_percent = st.text_input('MDVP:Jitter(%)')
            
    #     with col5:
    #         Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
            
    #     with col1:
    #         RAP = st.text_input('MDVP:RAP')
            
    #     with col2:
    #         PPQ = st.text_input('MDVP:PPQ')
            
    #     with col3:
    #         DDP = st.text_input('Jitter:DDP')
            
    #     with col4:
    #         Shimmer = st.text_input('MDVP:Shimmer')
            
    #     with col5:
    #         Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
            
    #     with col1:
    #         APQ3 = st.text_input('Shimmer:APQ3')
            
    #     with col2:
    #         APQ5 = st.text_input('Shimmer:APQ5')
            
    #     with col3:
    #         APQ = st.text_input('MDVP:APQ')
            
    #     with col4:
    #         DDA = st.text_input('Shimmer:DDA')
            
    #     with col5:
    #         NHR = st.text_input('NHR')
            
    #     with col1:
    #         HNR = st.text_input('HNR')
            
    #     with col2:
    #         RPDE = st.text_input('RPDE')
            
    #     with col3:
    #         DFA = st.text_input('DFA')
            
    #     with col4:
    #         spread1 = st.text_input('spread1')
            
    #     with col5:
    #         spread2 = st.text_input('spread2')
            
    #     with col1:
    #         D2 = st.text_input('D2')
            
    #     with col2:
    #         PPE = st.text_input('PPE')
            
        
        
    #     # code for Prediction
    #     parkinsons_diagnosis = ''
        
    #     # creating a button for Prediction    
    #     if st.button("Parkinson's Test Result"):
    #         parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
            
    #         if (parkinsons_prediction[0] == 1):
    #             parkinsons_diagnosis = "The person has Parkinson's disease"
    #         else:
    #             parkinsons_diagnosis = "The person does not have Parkinson's disease"
            
    #     st.success(parkinsons_diagnosis)


if selected2=="Report": 
        st.subheader("Analysis")
        with st.sidebar:
    
         selected = option_menu('Analysis',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction'
                           ],
                          icons=['activity','heart','person'],
                          default_index=0)
        if selected== 'Heart Disease Prediction':
       
        
        # df = pd.DataFrame(
        # [
        # {"command": "st.selectbox", "rating": 4, "is_widget": True},
        # {"command": "st.balloons", "rating": 5, "is_widget": False},
        # {"command": "st.time_input", "rating": 3, "is_widget": True},
        # ]
        # )

        # # edited_df = st.experimental_data_editor(df, num_rows='dynamic')
        # st.experimental_data_editor(df, key="data_editor") # 👈 Set a key
        # st.write("Here's the session state:")
        # st.write(st.session_state["data_editor"]) # 👈 Access the edited data
        # data=st.session_state["data_editor"]
        # st.dataframe(df)
            import plotly.express as px
            import plotly.figure_factory as ff

            data = pd.read_csv('C:/Users/rnegi/Desktop/ML complete/MLApp/heart_disease_data (1).csv')
            x1=data['age']
            df=data['target']==1
        # Plot!
            df1=data[df]
            counts = df1['cp'].value_counts()

            import plotly.graph_objects as go
       

            X=data.iloc[:,:-1]
            y=data.iloc[:,-1]
            corr_matrix = X.corrwith(y)

            # Plot the correlation matrix using Plotly
            fig = go.Figure(data=go.Bar(x=corr_matrix.index, y=corr_matrix))
            fig.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Correlation Coefficient')
            st.plotly_chart(fig)

            fig1 = px.scatter(data, x='age', y='chol', color='target', hover_data=['sex', 'cp'], size='thalach')

# display the plot in Streamlit
            st.plotly_chart(fig1)

        if selected =='Diabetes Prediction':
            try:
                
                data=pd.read_csv("diabetes.csv")
                columns = st.multiselect("Select columns to graph", data.columns)

                # create a scatter plot using plotly
                fig = px.scatter(data, x=columns[0], y=columns[1], color="Outcome")

                # display the chart in streamlit
                st.plotly_chart(fig)
            except:
                st.subheader("Please two attributes ⬆️")
                
                
        
            X1=data.iloc[:,:-1]
            y1=data.iloc[:,-1]
            corr1=X1.corrwith(y1)
            fig2 = go.Figure(data=go.Bar(x=corr1.index, y=corr1))
            fig2.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Correlation Coefficient')
            st.plotly_chart(fig2)
            
            
            # import base64
            # file=pd.read_csv('my_file.csv')
            # def download_csv():
            #     csv = file.to_csv(index=False)
            #     b64 = base64.b64encode(csv.encode()).decode()
            #     print(type(b64))
            #     href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
            #     return href


            # st.markdown(download_csv(), unsafe_allow_html=True)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


   


