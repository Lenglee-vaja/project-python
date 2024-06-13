from Home import st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
st.set_page_config(page_title="Real-time prediction",layout="centered")
st.subheader("Real-time prediction")

#Retrieve data from the database
with st.spinner("Retrieving data from the database..."):
    redis_face_db = face_rec.retrive_data(name = "academy:register")
    st.dataframe(redis_face_db)
st.success("Data successfully retrieved")
#Time
waitTime = 10 # in seconds
setTime = time.time()
realTimePred = face_rec.RealTimePred()
#Real time prediction 
#Use streamlit webrtc

def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")#3 dimension numpy array
    # operation that you can perform on the array
    pred_img = realTimePred.face_prediction(img, redis_face_db, 'facial_features',["Name","Role"], thresh = 0.5)
    timeNow = time.time()
    diffTime = timeNow - setTime
    if diffTime >= waitTime:
        realTimePred.saveLogs_redis()
        setTime = time.time() #reset time
        print("Saved data to redis database")
    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(key="realTimePrediction", video_frame_callback=video_frame_callback,
rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)