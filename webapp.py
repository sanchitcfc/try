import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from viz1 import viz
from final import db1
from main import dbmain
import requests

    





def Page_4():
    st.subheader("Page 4")
    sub_page = st.sidebar.selectbox("Select a sub-page", ["Sub-page 1", "Sub-page 2"])
    if sub_page == "Sub-page 1":
        st.write("This is sub-page 1.")
    elif sub_page == "Sub-page 2":
        st.write("This is sub-page 2.")


def Page_5():
    st.subheader("Page 5")
    sub_page = st.sidebar.selectbox("Select a sub-page", ["Sub-page 1", "Sub-page 2"])
    if sub_page == "Sub-page 1":
        st.write("This is sub-page 1.")
    elif sub_page == "Sub-page 2":
        st.write("This is sub-page 2.")



def main():
    st.set_page_config(page_title="Malware Mavericks", page_icon=":guardsman:", layout="wide")
    st.title("Malware Mavericks")
    # Adding a logo
    logo_url = 'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=634&q=80'
    st.sidebar.image(logo_url, width=50)
    pages = {"Visualize Data":viz,"DB_Chris":db1,"DB_Online":dbmain,"Page 4":Page_4,"Page 5":Page_5}
    choice = st.sidebar.selectbox("Select a page", list(pages.keys()))
    pages[choice]()

if __name__ == '__main__':
    main()
