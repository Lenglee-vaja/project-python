import streamlit as st

st.set_page_config(page_title="Attendance System", page_icon="📝", layout="wide")
st.header("Attendance System using face recognition") 

with st.spinner("Loading Models and Connecting to Redis..."):
    import face_rec
st.success("Models loaded successfully")

st.success("Connecting to Redis...")