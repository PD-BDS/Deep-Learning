import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from crew import CustomCrew

app = FastAPI(
    title="Crew AI Agent for Data Analysis ",
    description="API for running SQL data analysis using CrewAI",
    version="1.0.0",
)

class CrewRequest(BaseModel):
    query: str

class CrewResponse(BaseModel):
    result: str

@app.post("/run_analysis", response_model=CrewResponse)
async def run_crew(crew_request: CrewRequest):
    try:
        custom_crew = CustomCrew(crew_request.query)
        result = custom_crew.run()
        return CrewResponse(result=result.raw or str(result))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the application for Data Analysis"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)
