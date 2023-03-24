from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
import numpy as np
import pickle
import streamlit as st
from ML_models import *

# loading the saved model
LR_model = pickle.load(open('model_lr.pkl', 'rb'))

KNN_model = pickle.load(open('model_knn.pkl','rb'))

SVML_model = pickle.load(open('model_svml.pkl','rb'))

SVMK_model = pickle.load(open('model_svmk.pkl','rb'))


# creating a function for Prediction

def LR_prdeiction(input_data):
    
    lr_prediction = LR_model.predict([input_data])
    

    if (lr_prediction[0] == 0):
      return 'Line of Sight Signal'
    else:
      return 'Multipath Signal'

def KNN_prdeiction(input_data):
    
    knn_prediction = KNN_model.predict([input_data])
    

    if (knn_prediction[0] == 0):
      return "Line of Sight SIgnal"
    else:
      return "Multipath SIgnal"
  
def SVML_prdeiction(input_data):
    
    svml_prediction = SVML_model.predict([input_data])
    

    if (svml_prediction[0] == 0):
      return "Line of Sight SIgnal"
    else:
      return "Multipath SIgnal"

def SVMK_prdeiction(input_data):
    
    svmk_prediction = SVMK_model.predict([input_data])
    

    if (svmk_prediction[0] == 0):
      return "Line of Sight SIgnal"
    else:
      return "Multipath SIgnal"   
  
def main():
    
    
    # giving a title
    st.title('Multipath Error Detection')

    # getting the input data from the user
    
    carrier_noise_ratio = st.number_input('Carrier to Noise ratio value (dB-Hz)')
    pseudorange = st.number_input('Pseudorange (meters)')
    elevation_angle = st.number_input('Elevation Angle value (degrees)')
    
    # num_array = str_array.astype(int)
    # code for Prediction
    LR_pred_value = -1
    KNN_pred_value = -1
    SVML_pred_value = -1
    SVMK_pred_value = -1
    # creating a button for Prediction
    
    if st.button('Predict Error'):
        LR_pred_value = LR_prdeiction([carrier_noise_ratio, pseudorange, elevation_angle])
        KNN_pred_value = KNN_prdeiction([carrier_noise_ratio, pseudorange, elevation_angle])
        SVML_pred_value = SVML_prdeiction([carrier_noise_ratio, pseudorange, elevation_angle])
        SVMK_pred_value = SVMK_prdeiction([carrier_noise_ratio, pseudorange, elevation_angle])


        heading = "<h2>Logistic Regression</h2>"
        st.markdown(heading, unsafe_allow_html=True)
        if LR_pred_value == "Line of Sight Signal":
            st.success(LR_pred_value)
        else:
            st.warning(LR_pred_value)
        cm_html = "<h4>Confusion Matrix of Logistic Regression</h4>"
        cm_html += "<table><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>"
        cm_html += "<tr><th>Actual 0</th><td>{}</td><td>{}</td></tr>".format(cm_LR[0,0], cm_LR[0,1])
        cm_html += "<tr><th>Actual 1</th><td>{}</td><td>{}</td></tr>".format(cm_LR[1,0], cm_LR[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)

        heading = "<h2>K Nearest Neighbours</h2>"
        st.markdown(heading, unsafe_allow_html=True)
        if KNN_pred_value == "Line of Sight Signal":
            st.success(KNN_pred_value)
        else:
            st.warning(KNN_pred_value)
        cm_html = "<h4>Confusion Matrix of K Nearest Neighbours</h4>"
        cm_html += "<table><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>"
        cm_html += "<tr><th>Actual 0</th><td>{}</td><td>{}</td></tr>".format(cm_KNN[0,0], cm_KNN[0,1])
        cm_html += "<tr><th>Actual 1</th><td>{}</td><td>{}</td></tr>".format(cm_KNN[1,0], cm_KNN[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)

        heading = "<h2>SVM Linear </h2>"
        st.markdown(heading, unsafe_allow_html=True)
        if SVML_pred_value == "Line of Sight Signal":
            st.success(SVML_pred_value)
        else:
            st.warning(SVML_pred_value)
        cm_html = "<h4>Confusion Matrix of SVM Linear</h4>"
        cm_html += "<table><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>"
        cm_html += "<tr><th>Actual 0</th><td>{}</td><td>{}</td></tr>".format(cm_SVML[0,0], cm_SVML[0,1])
        cm_html += "<tr><th>Actual 1</th><td>{}</td><td>{}</td></tr>".format(cm_SVML[1,0], cm_SVML[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)
        
        heading = "<h2>SVM Kernel </h2>"
        st.markdown(heading, unsafe_allow_html=True)
        if SVMK_pred_value == "Line of Sight Signal":
            st.success(SVMK_pred_value)
        else:
            st.warning(SVMK_pred_value)
        cm_html = "<h4>Confusion Matrix of SVM using Kernel</h4>"
        cm_html += "<table><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr>"
        cm_html += "<tr><th>Actual 0</th><td>{}</td><td>{}</td></tr>".format(cm_SVMR[0,0], cm_SVMR[0,1])
        cm_html += "<tr><th>Actual 1</th><td>{}</td><td>{}</td></tr>".format(cm_SVMR[1,0], cm_SVMR[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)
        
    print()
    
    
if __name__ == '__main__':
    main()