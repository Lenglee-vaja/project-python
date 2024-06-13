from Home import st
from Home import face_rec
st.set_page_config(page_title="Reporting", layout="wide")
st.subheader("Reporting")

#extract data from redis to report
name = "academy:logs"
def load_logs(name,end = -1):
    logs_list = face_rec.r.lrange(name,0,end)
    return logs_list
#tabs to show info
tab1, tab2 = st.tabs([
    "Registered Data", "logs"
])
with tab1:
    if st.button("Refresh Data"):
    #Retrieve data from the database
        with st.spinner("Retrieving data from the database..."):
            redis_face_db = face_rec.retrive_data(name = "academy:register")
            st.dataframe(redis_face_db[["Name","Role"]])
with tab2:
    if st.button("Refresh logs"):
        st.write(load_logs(name=name))
