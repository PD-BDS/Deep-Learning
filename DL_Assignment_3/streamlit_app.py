import streamlit as st
import requests

# Streamlit app title
st.title("Dataset Analysis App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=["csv"])

# Button to run analysis
if st.button("Analyze Dataset") and uploaded_file:
    with st.spinner("Analyzing dataset..."):
        try:
            # Send file to FastAPI backend
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            response = requests.post("http://127.0.0.1:8000/upload_csv", files=files)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("Analysis completed!")

                # Display Dataset Description
                st.subheader("Dataset Description")
                st.write(result["description_output"])

                # Display Data Cleaning Output
                st.subheader("Data Cleaning Output")
                st.write(result["cleaning_output"])

                # Display Exploratory Data Analysis
                st.subheader("Exploratory Data Analysis")
                st.write(result["eda_output"])

                # Display Generated Visualizations
                st.subheader("Generated Visualizations")
                for img_path in result.get("visualization_output", []):
                    st.image(img_path, use_column_width=True)

            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")