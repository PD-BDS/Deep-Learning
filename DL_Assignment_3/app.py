import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import pandas as pd
from crewai import Crew
from agents import CustomAgents
from tasks import CustomTasks

# Initialize FastAPI app
app = FastAPI(
    title="Dataset Analysis API",
    description="API for analyzing datasets using Crew AI agents",
    version="1.0.0",
)

# Define response schemas
class DatasetAnalysisResponse(BaseModel):
    description_output: str
    cleaning_output: str
    eda_output: str
    visualization_output: list  # List of image paths

# Define the CustomCrew class
class CustomCrew:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def generate_visualizations(self):
        import seaborn as sns
        import matplotlib.pyplot as plt
        import os

        image_paths = []
        output_dir = "static/visualizations"
        os.makedirs(output_dir, exist_ok=True)

        # Generate histograms for numeric columns
        for column in self.dataset.select_dtypes(include=['int', 'float']).columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(self.dataset[column], kde=True, bins=20)
            plt.title(f'Distribution of {column}')
            image_path = f"{output_dir}/{column}_histogram.png"
            plt.savefig(image_path)
            plt.close()
            image_paths.append(image_path)

        return image_paths

    def run(self) -> dict:
        agents = CustomAgents()
        tasks = CustomTasks()

        # Initialize custom agents
        describer = agents.dataset_describer()
        cleaner = agents.data_cleaner()
        eda_agent = agents.data_eda()
        visualizer = agents.data_visualizer()

        # Initialize custom tasks
        task1 = tasks.describe_dataset(describer, self.dataset)
        task2 = tasks.clean_dataset(cleaner, self.dataset)
        task3 = tasks.perform_eda(eda_agent, self.dataset)
        task4 = tasks.visualize_data(visualizer, self.dataset)

        # Create and run the crew
        crew = Crew(
            agents=[describer, cleaner, eda_agent, visualizer],
            tasks=[task1, task2, task3, task4],
            verbose=True,
        )

        # Run the crew and capture outputs
        crew_output = crew.kickoff()

        # Parse the outputs
        outputs = crew_output if isinstance(crew_output, tuple) else [crew_output]
        if len(outputs) < 4:
            outputs.extend(["No output available"] * (4 - len(outputs)))

        # Generate visualizations
        visualization_output = self.generate_visualizations()

        # Prepare structured output
        structured_output = {
            "description_output": str(outputs[0]),
            "cleaning_output": str(outputs[1]),
            "eda_output": str(outputs[2]),
            "visualization_output": visualization_output,
        }

        return structured_output

# Define the API endpoint
@app.post("/upload_csv", response_model=DatasetAnalysisResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and analyze it using Crew AI agents.
    """
    try:
        df = pd.read_csv(file.file)
        custom_crew = CustomCrew(df)
        result = custom_crew.run()

        return DatasetAnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dataset Analysis API. Use /docs for API documentation."}

# Entry point for running the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)