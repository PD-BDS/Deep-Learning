import sqlite3
from crewai import LLM
from crewai.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLDataBaseTool,
    QuerySQLCheckerTool,
)
import os
import re

# Regex to Extract Code Snippet in triple backticks
flexible_pattern = re.compile(
    r"```python\s*(.*?)(```|$)",  # match until '```' or end-of-string
    re.DOTALL | re.IGNORECASE
)

def extract_code_block(raw_text: str) -> str:
    """
    Finds the substring after '```python' up to either the next triple backticks or end of string.
    Returns that as the code snippet. If not found, returns empty string.
    """
    match = flexible_pattern.search(raw_text)
    if match:
        code_part = match.group(1)
        # Remove leftover triple backticks just in case
        code_part = code_part.replace("```", "").strip()
        return code_part
    return ""

# Database file
DATABASE_FILE = "temp_db.sqlite"

# Function to create a fresh database connection
def get_db_connection():
    return sqlite3.connect(DATABASE_FILE, check_same_thread=False)

# Ensure the database connection is always fresh
def execute_query(query):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return results
    finally:
        conn.close()

# Tool to list tables in the database
@tool("list_tables")
def list_tables_tool():
    """List the available tables in the database."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_FILE}")
        return ListSQLDatabaseTool(db=db).invoke("")
    except Exception as e:
        return f"Error listing tables: {str(e)}"

# Tool to fetch the schema of a given table
@tool("tables_schema")
def tables_schema_tool(tables: str):
    """Show schema & sample rows for the given tables (comma-separated)."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_FILE}")
        return InfoSQLDatabaseTool(db=db).invoke(tables)
    except Exception as e:
        return f"Error fetching schema: {str(e)}"

# Tool to execute a SQL query
@tool("execute_sql")
def execute_sql_tool(sql_query: str):
    """Execute a SQL query against the database and return the result."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_FILE}")
        return QuerySQLDataBaseTool(db=db).invoke(sql_query)
    except Exception as e:
        return f"Error executing SQL query: {str(e)}"

# Tool to check the SQL query for correctness
@tool("check_sql")
def check_sql_tool(sql_query: str):
    """Check if the SQL query is correct and return suggestions/fixes."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{DATABASE_FILE}")
        llm_checker = LLM(model="ollama/llama3.2", base_url="http://localhost:11434", temperature=0.2)
        query_checker_tool = QuerySQLCheckerTool(db=db, llm=llm_checker)
        return query_checker_tool.invoke({"query": sql_query})
    except Exception as e:
        return f"Error checking SQL query: {str(e)}"

# Tool to generate visualization code (using the agent)
@tool("generate_visualization")
def generate_visualization_tool(query: str, df):
    """
    Generate the visualization code based on the user query and DataFrame.
    The agent uses the query to generate a visualization in Plotly.
    """
    try:
        # Here you would need the agent to generate Plotly code.
        # For this example, we're just mocking the agent's response.
        visualization_code = f"""
        import plotly.express as px
        fig = px.bar()
        fig
        """
        return visualization_code
    except Exception as e:
        return f"Error generating visualization code: {str(e)}"
