import gc
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from sign_language import path_to_tensor as imgload
from PIL import Image

x, y, w, h = 50, 50, 100, 50
size = (64, 64)
st.title("NSL Manual Alphabet")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = st.camera_input("Take a picture")

def sign_predict(image):
    model = tf.keras.models.load_model("saved_model/my_model")
    x = imgload(image).astype('float32')/255
    probs = model.predict(x)
    preds = np.argmax(probs, axis=1)
    return chr(preds[0]+65)

def load_image(img_file_buffer):
    # To read image file buffer with OpenCV:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return cv2_img

if camera is not None:
    while True:
        cam_frame = load_image(camera)
        frame = cv2.resize(cam_frame, size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        to_pil = Image.fromarray(frame_rgb)
        pred = sign_predict(to_pil)
        print(pred)
        cv2.putText(cam_frame, pred, (x+10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 
                    3, (0, 0, 255))
        FRAME_WINDOW.image(cam_frame)
        
        #if camera.isOpened():
        if not run:
            #camera.release()
            st.write('Stopped')
            break


