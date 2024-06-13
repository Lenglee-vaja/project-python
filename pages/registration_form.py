import streamlit as st
import numpy as np
from Home import face_rec
import cv2
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Registration form", layout="centered")
st.subheader("Registration form")
## Init Registration form
registration_form = face_rec.RegistrationForm()

#step1: collect person and role
person_name = st.text_input(label="Enter your name",placeholder="First and last name")
role = st.selectbox(label="Select your role",options=("Student","Teacher"))

#step2: collect facial embedding of that person
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")
    reg_img, embedding = registration_form.get_embedding(img)
    #save data to local computer txt
    if embedding is not None:
        with open("face_embedding.txt",mode="ab") as f:
            np.savetxt(f,embedding)
    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registration", video_frame_callback=video_callback_func,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

#step3: push data to redis

if st.button("Register"):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} Registered successfully")
    elif return_val == "name_false":
        st.error("Name cannot be empty")
    elif return_val == "file_false":
        st.error("Embedding file not found")