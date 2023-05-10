import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE


def dbmain():
    st.write("Read Data")
    df = pd.read_csv('sample_data.csv')

    st.write(df)

    df.Label.value_counts()

    st.write("Data sampling")

    df_minor = df[(df['Label']=='WebAttack')|(df['Label']=='Bot')|(df['Label']=='Infiltration')]
    df_BENIGN = df[(df['Label']=='BENIGN')]
    df_BENIGN = df_BENIGN.sample(n=None, frac=0.01, replace=False, weights=None, random_state=None, axis=0)
    df_DoS = df[(df['Label']=='DoS')]
    df_DoS = df_DoS.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
    df_PortScan = df[(df['Label']=='PortScan')]
    df_PortScan = df_PortScan.sample(n=None, frac=0.05, replace=False, weights=None, random_state=None, axis=0)
    df_BruteForce = df[(df['Label']=='BruteForce')]
    df_BruteForce = df_BruteForce.sample(n=None, frac=0.2, replace=False, weights=None, random_state=None, axis=0)

    df_s = df_BENIGN.append(df_DoS).append(df_PortScan).append(df_BruteForce).append(df_minor)



    df_s = df_s.sort_index()

    df_s.to_csv('sample2.csv',index=0)


    st.write("Preprocessing")

    df = pd.read_csv('sample2.csv')

    numeric_features = df.dtypes[df.dtypes != 'object'].index
    df[numeric_features] = df[numeric_features].apply(
        lambda x: (x - x.min()) / (x.max()-x.min()))

    df = df.fillna(0)


    st.write("Split train set and test set")

    labelencoder = LabelEncoder()
    df.iloc[:, -1] = labelencoder.fit_transform(df.iloc[:, -1])
    X = df.drop(['Label'],axis=1).values 
    y = df.iloc[:, -1].values.reshape(-1,1)
    y=np.ravel(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

    st.write(X_train.shape)

    pd.Series(y_train).value_counts()
    st.write(pd.Series(y_train).value_counts())


    st.write("Oversampling")
    smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})
    X_train, y_train = smote.fit_resample(X_train, y_train)
    pd.Series(y_train).value_counts()

    st.write(pd.Series(y_train).value_counts())


    st.write("ML model training")

    st.write("Decision Trees")
    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    dt_score=dt.score(X_test,y_test)
    y_predict=dt.predict(X_test)
    y_true=y_test
    st.write('Accuracy of DT: '+ str(dt_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of DT: '+(str(precision)))
    st.write('Recall of DT: '+(str(recall)))
    st.write('F1-score of DT: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()

    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)

    st.write("Random Forest")
    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train) 
    rf_score=rf.score(X_test,y_test)
    y_predict=rf.predict(X_test)
    y_true=y_test
    st.write('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of RF: '+(str(precision)))
    st.write('Recall of RF: '+(str(recall)))
    st.write('F1-score of RF: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)

    st.write("Extra Trees")

    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test)
    y_predict=et.predict(X_test)
    y_true=y_test
    st.write('Accuracy of ET: '+ str(et_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of ET: '+(str(precision)))
    st.write('Recall of ET: '+(str(recall)))
    st.write('F1-score of ET: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()

    et_train=et.predict(X_train)
    et_test=et.predict(X_test)




    st.write("Stacking Models")

    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        })
    base_predictions_train.head(5)

    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)

    x_train = np.concatenate(( dt_train, et_train, rf_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test), axis=1)

    stk = RandomForestClassifier().fit(x_train, y_train)

    y_predict=stk.predict(x_test)
    y_true=y_test
    stk_score=accuracy_score(y_true,y_predict)
    st.write('Accuracy of Stacking: '+ str(stk_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of Stacking: '+(str(precision)))
    st.write('Recall of Stacking: '+(str(recall)))
    st.write('F1-score of Stacking: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()


    st.write("Feature Selection")
    dt_feature = dt.feature_importances_
    rf_feature = rf.feature_importances_
    et_feature = et.feature_importances_

    avg_feature = (dt_feature + rf_feature + et_feature )/3

    feature=(df.drop(['Label'],axis=1)).columns.values
    st.write ("Features sorted by their score:")
    st.write (sorted(zip(map(lambda x: round(x, 3), avg_feature), feature), reverse=True))

    f_list = sorted(zip(map(lambda x: round(x, 3), avg_feature), feature), reverse=True)

    st.write(len(f_list))

    Sum = 0
    fs = []
    for i in range(0, len(f_list)):
        Sum = Sum + f_list[i][0]
        fs.append(f_list[i][1])
        if Sum>=0.9:
            break  

    X_fs = df[fs].values

    X_train, X_test, y_train, y_test = train_test_split(X_fs,y, train_size = 0.8, test_size = 0.2, random_state = 0,stratify = y)

    X_train.shape

    pd.Series(y_train).value_counts()


    st.write("OverSampling")

    smote=SMOTE(n_jobs=-1,sampling_strategy={4:1500})
    X_train, y_train = smote.fit_resample(X_train, y_train)
    pd.Series(y_train).value_counts()
    st.write(pd.Series(y_train).value_counts())

    dt = DecisionTreeClassifier(random_state = 0)
    dt.fit(X_train,y_train) 
    dt_score=dt.score(X_test,y_test)
    y_predict=dt.predict(X_test)
    y_true=y_test
    st.write('Accuracy of DT: '+ str(dt_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of DT: '+(str(precision)))
    st.write('Recall of DT: '+(str(recall)))
    st.write('F1-score of DT: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()


    dt_train=dt.predict(X_train)
    dt_test=dt.predict(X_test)

    rf = RandomForestClassifier(random_state = 0)
    rf.fit(X_train,y_train)
    rf_score=rf.score(X_test,y_test)
    y_predict=rf.predict(X_test)
    y_true=y_test
    st.write('Accuracy of RF: '+ str(rf_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of RF: '+(str(precision)))
    st.write('Recall of RF: '+(str(recall)))
    st.write('F1-score of RF: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()

    rf_train=rf.predict(X_train)
    rf_test=rf.predict(X_test)

    et = ExtraTreesClassifier(random_state = 0)
    et.fit(X_train,y_train) 
    et_score=et.score(X_test,y_test)
    y_predict=et.predict(X_test)
    y_true=y_test
    st.write('Accuracy of ET: '+ str(et_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of ET: '+(str(precision)))
    st.write('Recall of ET: '+(str(recall)))
    st.write('F1-score of ET: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()


    et_train=et.predict(X_train)
    et_test=et.predict(X_test)



    st.write("Stacking and Model Construction")

    base_predictions_train = pd.DataFrame( {
        'DecisionTree': dt_train.ravel(),
            'RandomForest': rf_train.ravel(),
        'ExtraTrees': et_train.ravel(),
        })
    base_predictions_train.head(5)

    dt_train=dt_train.reshape(-1, 1)
    et_train=et_train.reshape(-1, 1)
    rf_train=rf_train.reshape(-1, 1)
    dt_test=dt_test.reshape(-1, 1)
    et_test=et_test.reshape(-1, 1)
    rf_test=rf_test.reshape(-1, 1)

    x_train = np.concatenate(( dt_train, et_train, rf_train), axis=1)
    x_test = np.concatenate(( dt_test, et_test, rf_test), axis=1)

    stk = RandomForestClassifier().fit(x_train, y_train)
    y_predict=stk.predict(x_test)
    y_true=y_test
    stk_score=accuracy_score(y_true,y_predict)
    st.write('Accuracy of Stacking: '+ str(stk_score))
    precision,recall,fscore,none= precision_recall_fscore_support(y_true, y_predict, average='weighted') 
    st.write('Precision of Stacking: '+(str(precision)))
    st.write('Recall of Stacking: '+(str(recall)))
    st.write('F1-score of Stacking: '+(str(fscore)))
    st.write(classification_report(y_true,y_predict))
    cm=confusion_matrix(y_true,y_predict)
    f,ax=plt.subplots(figsize=(5,5))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("y_pred")
    plt.ylabel("y_true")
    st.pyplot()
