from crewai import Task
from textwrap import dedent

class CustomTasks:
    def describe_dataset(self, agent, dataset):
        return Task(
            description=dedent(f"""
                Analyze the dataset and summarize its structure, column descriptions, and key insights.
                Dataset Preview: {dataset.head().to_string()}
            """),
            expected_output="A structured summary of the dataset and its features.",
            agent=agent,
        )

    def clean_dataset(self, agent, dataset):
        return Task(
            description="Perform data cleaning, including handling missing values and format issues. Return the cleaned dataset and techniques used.",
            expected_output="A cleaned dataset with preprocessing details.",
            agent=agent,
        )

    def perform_eda(self, agent, cleaned_dataset):
        return Task(
            description="Analyze the cleaned dataset and generate exploratory data analysis insights.",
            expected_output="EDA report with statistical summaries and key insights.",
            agent=agent,
        )

    def visualize_data(self, agent, cleaned_dataset):
        return Task(
            description="Generate actual visualization images (plots) using Matplotlib and Seaborn. Save them as image files.",
            expected_output="Paths to the generated visualization images.",
            agent=agent,
        )
