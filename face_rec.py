import pandas as pd
import numpy as np
import cv2
import redis
import os
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time
from datetime import datetime

#connect to redis database
host = "redis-18912.c16.us-east-1-3.ec2.redns.redis-cloud.com"
port = 18912
password = "wKCy3CefJ6RAHpeTSi3iRFlNPxxkZqQf"

r = redis.Redis(
    host=host,
    port=port,
    password=password,
)
# retrive data from redis
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x, dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(),index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ["name_role", 'facial_features']
    retrive_df[["Name","Role"]] = retrive_df["name_role"].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[["Name","Role","facial_features"]]
# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                      root='insightface_model',
                      providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0, det_size=(640, 640),  det_thresh=0.5)

# ml search algrithem
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=["Name","Role"],thresh=0.5):
    """
    consine similarity base search algorithm
    """
    # step 1: take the dataframe (collection of data)
    dataframe = dataframe.copy()
    #step 2: Index face embeding from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)
    #step 3: cal. cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe["cosine"] = similar_arr
     #step 4: filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        data_filter.reset_index(drop = True, inplace = True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = "Unknown"
        person_role = "Unknown"
    return person_name, person_role
## Real time prediction
# We need to save logs for every 1 minute
class RealTimePred:
    def __init__(self):
        self.logs = dict(name = [], role = [], current_time = [])
    def reset_dict(self):
        self.logs = dict(name = [], role = [], current_time = [])
    def saveLogs_redis(self):
        #step1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        #step2: drop the duplicate information (distinct name) 
        dataframe.drop_duplicates('name', inplace= True)
        #step3: push data to redis
        #encode the data
        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []

        for name, role, ctime in zip(name_list, role_list, ctime_list):
            if name != "Unknown":
                concat_string = f"{name}@{role}@{ctime}"  
                encoded_data.append(concat_string)
                print(encoded_data)
        if len(encoded_data) > 0:
            r.lpush("academy:logs", *encoded_data)
        self.reset_dict()
    def face_prediction(self,image_test,dataframe,feature_column,name_role=["Name","Role"],thresh=0.5):
        # step1: find the time
        current_time = str(datetime.now())
        # step1: take the test image and apply to insightface
        results = faceapp.get(image_test)
        test_copy = image_test.copy()
        # step2: use loop the extract each embedding and past to ml_search_algorithem
        for res in results:
            x1, y1, x2, y2 = res["bbox"].astype(int)
            embeddings = res["embedding"]
            person_name, person_role = ml_search_algorithm(dataframe,feature_column,test_vector=embeddings, name_role =name_role,thresh = thresh )
            if person_name == "Unknown":
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,(color),2)
            #save info in logs dict
            self.logs["name"].append(person_name)
            self.logs["role"].append(person_role)
            self.logs["current_time"].append(current_time)
        return test_copy

#Registration form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0
    def get_embedding(self,frame):
        #get result from insight face model
        results = faceapp.get(frame, max_num=1)
        embeddings = None
        for res in results:
            self.sample +=1
            x1, y1, x2, y2 = res["bbox"].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            # put text simple info
            text = f"samples: {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            #facial feature
            embeddings = res["embedding"]
        return frame,embeddings
    def save_data_in_redis_db(self, name, role):
        #validation name
        if name is not None:
            if name.strip() != "":
                key = f'{name}@{role}'
            else:
                return 'name_false'
        else: 
            return 'name_false'
        # if face_embedding is exist
        if 'face_embedding.txt' not in os.listdir():
            return "file_false"
        #step1: Load "face_embedding.txt"
        x_array = np.loadtxt("face_embedding.txt", dtype=np.float32)
        #step2: convert into numpy array
        received_simples = int(x_array.size/512)
        x_array = x_array.reshape(received_simples,512)
        x_array = np.asarray(x_array)

        #step3: cal. mean embedding
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        #step4: push data to redis
        #redis hashes
        r.hset(name="academy:register",key=key,value=x_mean_bytes)
        os.remove("face_embedding.txt")
        self.reset()
        return True