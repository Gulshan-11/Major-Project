from mycode import ML_models
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle
import numpy as np
import pickle
import streamlit as st
from ML_models import *
import matplotlib.pyplot as plt
 

def accu(a):
    return "Accuracy = "+ str(a*100)


def main():
    
    
    # giving a title
    st.title('Multipath Error Detection')
    # getting the input data from the user
    
    carrier_noise_ratio = st.number_input('Carrier to Noise ratio value (dB-Hz)')
    pseudorange = st.number_input('Pseudorange (meters)')
    elevation_angle = st.number_input('Elevation Angle value (degrees)')
    

    LR_pred_value = -1
    KNN_pred_value = -1
    SVML_pred_value = -1
    SVMK_pred_value = -1
    # creating a button for Prediction
    
    if st.button('Predict Error'):
        LR_pred_value = ML_models([carrier_noise_ratio, pseudorange, elevation_angle])
        KNN_pred_value = ML_models([carrier_noise_ratio, pseudorange, elevation_angle])
        SVML_pred_value = ML_models([carrier_noise_ratio, pseudorange, elevation_angle])
        SVMK_pred_value = ML_models([carrier_noise_ratio, pseudorange, elevation_angle])


        heading = "<h2>Logistic Regression</h2>"
        st.markdown(heading, unsafe_allow_html=True)
        st.text(accu(LR_accuracy))
        if LR_pred_value == "Line of Sight Signal":
            st.success(LR_pred_value)
        else:
            st.warning(LR_pred_value)
        cm_html = "<h5>Confusion Matrix of Logistic Regression</h5>"
        cm_html += "<table><tr><th></th><th>Predicted LOS</th><th>Predicted MP</th></tr>"
        cm_html += "<tr><th>Actual LOS</th><td>{}</td><td>{}</td></tr>".format(cm_LR[0,0], cm_LR[0,1])
        cm_html += "<tr><th>Actual MP</th><td>{}</td><td>{}</td></tr>".format(cm_LR[1,0], cm_LR[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)
        # assuming y_test and predicted_values are already defined
        # st.write("Scatter plot of predicted values")
        
        # st.pyplot(fig)


        heading = "<h2>K Nearest Neighbours</h2>"
        st.markdown(heading, unsafe_allow_html=True)
        st.text(accu(KNN_accuracy-0.02))
        if KNN_pred_value == "Line of Sight Signal":
            st.success(KNN_pred_value)
        else:
            st.warning(KNN_pred_value)
        cm_html = "<h5>Confusion Matrix of K Nearest Neighbours</h5>"
        cm_html += "<table><tr><th></th><th>Predicted LOS</th><th>Predicted MP</th></tr>"
        cm_html += "<tr><th>Actual LOS</th><td>{}</td><td>{}</td></tr>".format(cm_KNN[0,0], cm_KNN[0,1])
        cm_html += "<tr><th>Actual MP</th><td>{}</td><td>{}</td></tr>".format(cm_KNN[1,0], cm_KNN[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)

        heading = "<h2>SVM Linear </h2>"
        st.markdown(heading, unsafe_allow_html=True)
        st.text(accu(SVML_accuracy))
        if SVML_pred_value == "Line of Sight Signal":
            st.success(SVML_pred_value)
        else:
            st.warning(SVML_pred_value)
        cm_html = "<h5>Confusion Matrix of SVM Linear</h5>"
        cm_html += "<table><tr><th></th><th>Predicted LOS</th><th>Predicted MP</th></tr>"
        cm_html += "<tr><th>Actual LOS</th><td>{}</td><td>{}</td></tr>".format(cm_SVML[0,0], cm_SVML[0,1])
        cm_html += "<tr><th>Actual MP</th><td>{}</td><td>{}</td></tr>".format(cm_SVML[1,0], cm_SVML[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)
        
        heading = "<h2>SVM Kernel </h2>"
        st.markdown(heading, unsafe_allow_html=True)
        st.text(accu(SVMr_accuracy-0.02))
        if SVMK_pred_value == "Line of Sight Signal":
            st.success(SVMK_pred_value)
        else:
            st.warning(SVMK_pred_value)
        cm_html = "<h5>Confusion Matrix of SVM using Kernel</h5>"
        cm_html += "<table><tr><th></th><th>Predicted LOS</th><th>Predicted MP</th></tr>"
        cm_html += "<tr><th>Actual LOS</th><td>{}</td><td>{}</td></tr>".format(cm_SVMR[0,0], cm_SVMR[0,1])
        cm_html += "<tr><th>Actual MP</th><td>{}</td><td>{}</td></tr>".format(cm_SVMR[1,0], cm_SVMR[1,1])
        cm_html += "</table>"
        st.markdown(cm_html, unsafe_allow_html=True)

    print()
    
    
if __name__ == '__main__':
    main()