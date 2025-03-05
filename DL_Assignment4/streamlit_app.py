import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from agents import CustomAgents
from tasks import CustomTasks
from crewai import Crew, Process
from custom_tools import extract_code_block 

# Page config
st.set_page_config(layout="wide")
st.title("AI Agent For Data Analysis")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV for analysis", type="csv")
if not uploaded_file:
    st.info("Please upload a CSV file to begin.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(df.head())

# Database file
db_file = "temp_db.sqlite"

# Function to initialize the database safely
def init_database():
    if os.path.isfile(db_file):
        os.remove(db_file)  # Ensure old DB is removed to prevent locking issues

    conn = sqlite3.connect(db_file, check_same_thread=False)
    try:
        df.to_sql(name="data_table", con=conn, if_exists="replace", index=False)
    finally:
        conn.close()  # Always close the connection after initializing the DB

if "db_initialized" not in st.session_state:
    st.session_state["db_initialized"] = False

if not st.session_state["db_initialized"]:
    init_database()
    st.session_state["db_initialized"] = True
    st.success("Database created from uploaded CSV.")

# Ask the Agent
st.subheader("Ask The Agent About The Dataset")
user_query = st.text_input("Example: 'Show the average of a numeric column'", "")

# Generate Report
if st.button("Generate Report"):
    if not user_query.strip():
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner("Running the multi-agent pipeline..."):
        try:
            response = requests.post("http://127.0.0.1:8002/run_analysis", json={"query": user_query})
            if response.status_code == 200:
                result = response.json()
                st.session_state["report_result"] = result["result"]  # Store the report result in session state
                st.success("Analysis Complete!")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")

# Display the generated report if it exists
if "report_result" in st.session_state:
    st.text_area("Generated Report", st.session_state["report_result"], height=500)

# Data Visualization Section
st.subheader("Visualize the Data")
viz_prompt = st.text_area(
    "Write your instructions. Example:\n'Please create a bar chart of average Score by Subject using the data'"
)

# Inside the "Generate Plot" button block:
if st.button("Generate Plot"):
    if not viz_prompt.strip():
        st.warning("Please enter a visualization prompt.")
        st.stop()
    
    # Save the DataFrame as a temporary CSV for the agent's reference
    csv_path = "temp.csv"
    df.to_csv(csv_path, index=False)

    # Create agent's prompt to generate code for the visualization
    agent_prompt = f"""
        We have a CSV file named 'temp.csv' with columns: {', '.join(df.columns)}.
        Below is the dataset:
        {df.to_csv(index=False)}

        User wants a Plotly figure: "{viz_prompt}"

        Produce the code in triple backticks.
    """

    # Instantiate agents and tasks
    agents = CustomAgents()
    tasks = CustomTasks()

    # Create the Data Visualization Agent task
    data_viz_agent = agents.data_visualization_agent()
    data_viz_task = tasks.generate_visualization(data_viz_agent, agent_prompt, df)

    # Execute the crew process
    viz_crew = Crew(
        agents=[data_viz_agent],
        tasks=[data_viz_task],
        process=Process.sequential,
        verbose=True,
        output_log_file="crew.log"
    )

    with st.spinner("Generating visualization..."):
        crew_output = viz_crew.kickoff(inputs={"query": viz_prompt})
        fig_code = extract_code_block(crew_output.raw)  # Using the updated extract_code_block function
        if fig_code:
            try:
                # Execute the code and create the Plotly figure
                local_vars = {}
                exec(fig_code, {}, local_vars)  # Execute and store the output in local_vars
                fig = local_vars.get('fig')  # Get the figure object from the local variables

                if fig:
                    st.session_state["fig_result"] = fig  # Store the plot in session state
                    st.success("Plot generated successfully!")
                else:
                    st.error("No valid figure object was created.")
            except Exception as e:
                st.error(f"Error generating plot: {e}")

# Display the generated plot if it exists
if "fig_result" in st.session_state:
    st.plotly_chart(st.session_state["fig_result"])
