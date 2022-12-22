from multiprocessing.sharedctypes import Value
from os import X_OK
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss

st.markdown(
"""
# Nguyễn Văn Toàn
# Mssv: 20522028
## Principal Components Regression với Streamlit 
""")

uploaded_file = st.file_uploader("Dataset")

header = st.container()
features = st.container()
pca = st.container()
split = st.container()
f1_max = st.container()
output = st.container()
tuongquan = st.container()

# Upload du lieu
with header:
    X, y = load_wine(return_X_y=True)

def one_hot(x):
    classes= np.unique(x)
    one_hot_vectors = np.zeros((x.shape[0],len(classes)))
    for i, cls in enumerate(x):
        one_hot_vectors[i, np.where(classes == cls)] = 1
    return one_hot_vectors

with split:
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"))
    # train_test, k_fold = st.columns(2)
    if split_type == "Train-Test Split":
        rate_train = st.slider('Train', 0, 100, 0,1,format='%d%%')
    else:
        st.header("Numbers of Fold")
        k_fold = st.selectbox(" ", range(2, X.shape[0]))
        Kfold = KFold(n_splits=k_fold)

with pca:
    Input_feature = st.selectbox(" ", ("Principal Component Analysis", "No Principal Component Analysis"))

# Trộn data
if Input_feature == "Principal Component Analysis":
    X, y = shuffle(X, y, random_state=0)
    Run = st.button("Run")
    if Run:
        with f1_max:
            max_score = []
            st.header("MAX F1")
            for i in range(X.shape[1]):
                # Kfold = KFold(n_splits=5)
                f1_list = []
                for train_ids, test_ids in Kfold.split(X, y):
                    X_train = X[train_ids]
                    X_test = X[test_ids]
                    pca = PCA(n_components=int(i)+1)
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)
                    Reg = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10)
                    
                    Reg.fit(X_train, y[train_ids])
                    y_pred = Reg.predict(X_test)
                    f1_list.append(f1_score(y[test_ids],y_pred,average='macro'))
                f1 = np.asarray(f1_list)
                f1_list = pd.DataFrame(f1_list)
                max_score.append(sum(f1_list[0])/len(f1_list))
            k = np.arange(1,14,1)
            max_score = np.asarray(max_score)
            scores = pd.DataFrame({
                'Score':max_score,
                'K': k
            })
            # bar_chart = alt.Chart(scores).mark_bar().encode(
            #     y='Score',
            #     x='K'
            # )
            bar_chart = alt.Chart(scores).mark_line(color="Red").encode(
                y = 'Score',
                x = 'K'
            ).properties(width = 650, height = 500, title = "Line Plot").interactive()
            # st.write(scores.max)color="Yellow"
            st.altair_chart(bar_chart,use_container_width=True)
