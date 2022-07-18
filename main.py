import pandas as pd
import numpy as np
from psutil import users
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from collections import defaultdict
# def load_model():
#     with open("save_steps.pkl", "rb") as file:
#         data = pickle.load(file)
#     return data
st.title("Fraud Risk Management")

xls = pd.ExcelFile(r"C:\Users\Admin\Desktop\NEW1.xlsx")   
users = ['USER_1','USER_2','USER_3','USER_4','USER_5','USER_6','USER_7','USER_8','USER_9','USER_10']
val = st.selectbox("Select the User",users)
df = pd.read_excel(xls,val)


df.dropna(subset = ['Request Date Time'],inplace = True)

def fnc(val):
    x = str(val)
    if ':' in x[11:13]:
        return int(x[11:12])
    return int(x[11:13])
    
df['Request Date Time']=df['Request Date Time'].apply(fnc)

labelencoder=LabelEncoder()

df['Payer Account Number ( for DEBIT)']=labelencoder.fit_transform(df['Payer Account Number ( for DEBIT)'])
df['PSP DEVICE']=labelencoder.fit_transform(df['PSP DEVICE'])
df.drop('Payer Account Number ( for DEBIT)', axis=1, inplace=True)
total_tran = len(df.index)

x = df.copy()


kmeans = KMeans(3)
# Fit the data
kmeans.fit(x)
# Create a copy of the input data
clusters = x.copy()
# Take note of the predicted clusters 













clusters['cluster_pred']=kmeans.fit_predict(x)

# Plot the data using the Annual Income and the Spending Score
plt.scatter(clusters['Amount'],clusters['Request Date Time'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Amount')
plt.ylabel('Response Date Time')

st.dataframe(df)
st.title(f"Clusters of {val}")
st.pyplot(plt)


cl0 = clusters[clusters['cluster_pred'] == 0]
cl1 = clusters[clusters['cluster_pred'] == 1]
cl2 = clusters[clusters['cluster_pred'] == 2]
len1 = len(cl0.index)
len2 = len(cl1.index)
len3 = len(cl2.index)

values = defaultdict()
minval = min(cl0['Amount'])
maxval = max(cl0['Amount'])
mintran = min(cl0['Request Date Time'])
maxtran = max(cl0['Request Date Time'])
values[0] = {"minval":minval,"maxval":maxval,"mintran":mintran,"maxtran":maxtran,"perc":(len1/total_tran)*100}


minval = min(cl1['Amount'])
maxval = max(cl1['Amount'])
mintran = min(cl1['Request Date Time'])
maxtran = max(cl1['Request Date Time'])
values[1] = {"minval":minval,"maxval":maxval,"mintran":mintran,"maxtran":maxtran,"perc":(len2/total_tran)*100}


minval = min(cl2['Amount'])
maxval = max(cl2['Amount'])
mintran = min(cl2['Request Date Time'])
maxtran = max(cl2['Request Date Time'])
values[2] = {"minval":minval,"maxval":maxval,"mintran":mintran,"maxtran":maxtran,"perc":(len3/total_tran)*100}

values['devices'] = list(df['PSP DEVICE'].unique())




#donot touch below

st.title("For Testing Purpose")
hrs = np.arange(0,24,dtype=int)
device = values['devices']

amount = st.slider("Select the amount", min_value = 1, max_value=100000,value = 300)
time = st.selectbox("Time of the day",hrs)
dev = device[st.selectbox("Device used",device)]

st.write(amount,time,dev)
thresh = st.selectbox("Select the threshold value", [2.5,3,3.5,4,4.5,5])
ok = st.button("Predict the cluster",help="Uses the model to predict the flags")

pred_val = [time,amount,dev]
if ok:
    x=kmeans.predict([pred_val])
    # st.write(x)
    minval = values[x[0]]["minval"]
    maxval = values[x[0]]["maxval"]
    mintran = values[x[0]]["mintran"]
    maxtran = values[x[0]]["maxtran"]

    # st.write(minval,maxval,mintran,maxtran)

    
    # if values[x[0]]["perc"] <= thresh:
    if (pred_val[0] >= mintran) and (pred_val[0] <= maxtran):
        st.write("Flag not Raised")
    else:
        st.write('Flag Raised')
    # else:
    #     st.write("allow the transaction")
