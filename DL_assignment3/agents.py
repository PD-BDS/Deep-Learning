from crewai import Agent, LLM
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

class CustomAgents:
    def __init__(self):
        # Use Llama 3.2 locally
        self.Llama = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

    def dataset_describer(self):
        return Agent(
            role="Dataset Describer",
            backstory="An AI that summarizes dataset structure and features.",
            goal="Provide a structured summary of the dataset and its features.",
            llm=self.Llama,
        )

    def data_cleaner(self):
        return Agent(
            role="Data Cleaner",
            backstory="An AI expert in cleaning and preprocessing datasets.",
            goal="Identify and handle missing values, inconsistencies, and formatting issues. Pass cleaned dataset forward.",
            llm=self.Llama,
        )

    def data_eda(self):
        return Agent(
            role="EDA Analyst",
            backstory="A data scientist skilled in exploratory data analysis.",
            goal="Analyze cleaned dataset, generate insights, and highlight important patterns.",
            llm=self.Llama,
        )

    def data_visualizer(self):
        return Agent(
            role="Data Visualization Expert",
            backstory="An AI agent generating visualizations from given dataset.",
            goal="Generate 2-3 visualizations and provide key insights.",
            llm=self.Llama,
        )
