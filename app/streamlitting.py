
import gc
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from sign_language import path_to_tensor as imgload


#Enable garbage collection
gc.enable()

#Hide warnings
st.set_option("deprecation.showfileUploaderEncoding", False)

#Set App title
st.title('Nigerian Sign Language Recognition Web App')


#Set the directory path
my_path= '.'

img_1_path= my_path + '/images/img_1.jpg'
img_2_path= my_path + '/images/img_2.jpg'
img_3_path= my_path + '/images/img_3.jpg'


#Read and display the banner
#st.sidebar.image(banner_path,use_column_width=True)1

#App description
st.write("The app recognises the manual alphabet that is gestured in the image. The model was trained with tensorflow transfer learning.")
st.write('**For more info, Code is at:** [Github repository](https://github.com/Amiiney/cld-app-streamlit) **|**')
st.markdown('***')




#Set the selectbox for demo images
st.write('**Select an image for a DEMO**')
menu = ['Select an Image','Image 1', 'Image 2', 'Image 3']
choice = st.selectbox('Choose an image', menu)


#Set the box for the user to upload an image
st.write("**Upload your Image**")
uploaded_image = st.file_uploader("Upload your image in JPG or PNG format", type=["jpg", "png"])

def sign_predict(image):
    model = tf.keras.models.load_model("saved_model/my_model")
    x = imgload(image).astype('float32')/255
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    return chr(preds[0]+65)

#Function to deploy the model and print the report
def deploy(file_path=None,uploaded_image=uploaded_image, uploaded=False, demo=True):
    
    #Display the uploaded/selected image
    st.markdown('***')
    st.markdown("Model is now Predicting", unsafe_allow_html=True)
    
    st.sidebar.markdown("image_uploaded_successfully", unsafe_allow_html=True)
    st.sidebar.image(file_path, width=301, channels='BGR')
    st.sidebar.markdown("**MODEL PREDICTED is: '[ {} ]'**".format(sign_predict(file_path)))


#Set red flag if no image is selected/uploaded
if uploaded_image is None and choice=='Select an Image':
    st.sidebar.markdown("app_off", unsafe_allow_html=True)
    st.sidebar.markdown("No uploaded Image", unsafe_allow_html=True)


#Deploy the model if the user uploads an image
if uploaded_image is not None:
    #Close the demo
    choice='Select an Image'
    #Deploy the model with the uploaded image
    deploy(uploaded_image, uploaded=True, demo=False)
    del uploaded_image


#Deploy the model if the user selects Image 1
if choice== 'Image 1':
    deploy(img_1_path)


#Deploy the model if the user selects Image 2
if choice== 'Image 2':
    deploy(img_2_path) 


#Deploy the model if the user selects Image 3
if choice== 'Image 3':
    deploy(img_3_path)  


