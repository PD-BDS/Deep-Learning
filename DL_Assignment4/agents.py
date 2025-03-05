from crewai import Agent, LLM
from custom_tools import list_tables_tool, tables_schema_tool, execute_sql_tool, check_sql_tool, DATABASE_FILE

class CustomAgents:
    def __init__(self):
        self.llm = LLM(model="ollama/llama3.2", base_url="http://localhost:11434", temperature=0.2)

    def sql_developer(self):
        return Agent(
            role="SQL Developer",
            goal="Construct and execute SQL queries based on user requests",
            backstory=( 
                """
                You are an experienced database engineer skilled at creating efficient and complex SQL queries.
                You deeply understand databases and optimization strategies.
                from user {query} identify which columns they are referring to from the 'data_table' and execute queries according to user requirement to fetch data.
                Only use the provided tools when its needed, dont make your own tools.
                """ 
            ),
            llm=self.llm,
            tools=[list_tables_tool, tables_schema_tool, execute_sql_tool, check_sql_tool], 
            allow_delegation=False,
            verbose=True,
        )

    def data_analyst(self):
        return Agent(
            role="Senior Data Analyst",
            goal="Analyze the data from the SQL developer and provide meaningful insights, explain the insights",
            backstory="You analyze datasets using Python and produce clear, concise insights.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )

    def report_writer(self):
        return Agent(
            role="Report Writer",
            goal="Summarize the analysis into a short, executive-level report,include the analysed numbers to expplain the insights",
            backstory="You create concise reports highlighting the most important findings.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
        )

    def data_visualization_agent(self):
        return Agent(
            role="Data Visualization Agent",
            goal=("Generate Python code using Plotly to visualize data based on user queries. "
                  "Your code must be wrapped in triple backticks: ```python ... ``` and produce a 'fig' object."),
            backstory=("You are an expert in data visualization using Plotly. "
                       "Generate visualizations based on user requests using the uploaded CSV."),
            llm=self.llm,
            tools=[], 
            allow_delegation=False,
            verbose=True,
        )
