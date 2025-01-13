import streamlit as st
import requests
import pandas as pd

# Streamlit app configuration
st.title("RAG-Enabled CRM System")
st.sidebar.header("Navigation")

# API Endpoint
API_URL = "http://127.0.0.1:8000/api"  # Backend API endpoint

# Sidebar and Main Application
st.sidebar.subheader("Steps")
uploaded_file = st.sidebar.file_uploader("Upload your text or CSV file", type=["txt", "csv"])
process_status = False

if uploaded_file:
    st.sidebar.write("File uploaded successfully!")
    st.sidebar.write("You can now use additional features.")
    process_status = True

# Check if data has been uploaded
if process_status:
    options = st.sidebar.radio(
        "Choose an option",
        ["Recommendation", "Real-Time Sentiment Analysis", "Object Handling", "Query with RAG"]
    )

    # Upload data
    if st.sidebar.button("Upload & Process"):
        if uploaded_file.name.endswith(".txt"):
            data = uploaded_file.read().decode("utf-8")
        elif uploaded_file.name.endswith(".csv"):
            data = "\n".join(pd.read_csv(uploaded_file)["text"].tolist())

        # Send data to the backend
        response = requests.post(f"{API_URL}/add-data", json={"data": data})
        if response.status_code == 200:
            st.success("Data uploaded and processed successfully!")
        else:
            st.error(f"Failed to upload data: {response.json().get('detail', 'Unknown error')}")

    # Feature: Recommendation
    if options == "Recommendation":
        st.subheader("Recommendations")
        query = st.text_input("Enter your query for recommendations:")
        if st.button("Get Recommendation"):
            response = requests.post(f"{API_URL}/recommendation", json={"query": query})
            if response.status_code == 200:
                response_data = response.json()
                st.write("### Recommendations:")
                st.write(response_data["recommendations"])
            else:
                st.error(f"Failed to fetch recommendations: {response.json().get('detail', 'Unknown error')}")

    # Feature: Real-Time Sentiment Analysis
    elif options == "Real-Time Sentiment Analysis":
        st.subheader("Real-Time Sentiment Analysis")
        text = st.text_area("Enter text for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            response = requests.post(f"{API_URL}/sentiment", json={"text": text})
            if response.status_code == 200:
                response_data = response.json()
                st.write("### Sentiment Analysis Result:")
                st.write(response_data["sentiment_analysis"])
            else:
                st.error(f"Failed to analyze sentiment: {response.json().get('detail', 'Unknown error')}")

    # Feature: Object Handling
    elif options == "Object Handling":
        st.subheader("Object Handling")
        object_input = st.text_input("Enter object data to handle:")
        if st.button("Handle Object"):
            response = requests.post(f"{API_URL}/handle-object", json={"object": object_input})
            if response.status_code == 200:
                response_data = response.json()
                st.write("### Object Handling Result:")
                st.write(response_data)
            else:
                st.error(f"Failed to handle object: {response.json().get('detail', 'Unknown error')}")

    # Feature: Query with RAG
    elif options == "Query with RAG":
        st.subheader("Query with RAG")
        query = st.text_area("Enter your query:")
        if st.button("Get Answer"):
            response = requests.post(f"{API_URL}/recommendation", json={"query": query})
            if response.status_code == 200:
                response_data = response.json()
                st.write("### Query:")
                st.write(query)
                st.write("### RAG Response:")
                st.write(response_data["recommendations"])
                st.write("### Retrieved Documents:")
                for doc in response_data.get("retrieved_documents", []):
                    st.write(f"- {doc['metadata'].get('text', '')}")
            else:
                st.error(f"Failed to fetch the RAG response: {response.json().get('detail', 'Unknown error')}")

else:
    st.write("Please upload a file to proceed.")
