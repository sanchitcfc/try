import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests


def viz():
    file_id = '1kttchmkJ54VxcpyDTpc65jGCz8VNl7Bb'  # Replace with your file ID
    confirm_url = f"https://drive.google.com/uc?id={file_id}&export=download"

# Path and name of the output file
    output = 'data.csv'

    session = requests.Session()
    response = session.get(confirm_url, stream=True)
    token = None

    if 'confirm' in response.content.decode():
        token_start = response.content.decode().find('confirm=')
        token_end = response.content.decode().find('&', token_start)
        if token_start != -1 and token_end != -1:
            token = response.content.decode()[token_start + 8:token_end]

# Download the file with confirmation token
    if token:
        download_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm={token}"
        response = session.get(download_url, stream=True)

    # Save the file
        with open(output, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(output)
    else:
        print("Unable to obtain confirmation token.")

    columns_to_visualize = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'Fwd Pkt Len Max', 'Bwd Pkt Len Max', 
        'Flow IAT Mean', 'Flow IAT Std', 'Fwd PSH Flags', 'Bwd PSH Flags',
        'Fwd URG Flags', 'Bwd URG Flags', 'SYN Flag Cnt', 'RST Flag Cnt',
        'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count',
        'ECE Flag Cnt'
    ]

    for column in columns_to_visualize:
        plt.figure()
        plt.title(f'{column} Distribution')
        df[column].plot(kind='hist', rwidth=0.8)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot()  


