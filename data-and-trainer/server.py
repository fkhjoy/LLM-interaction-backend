from fastapi import FastAPI
from data_trainer import UnityNLPPipeline
import json

app = FastAPI()
pipeline = UnityNLPPipeline()

def scene_command(user_query, objects):
    objs = list(objects.split(","))
    pipeline.update_scene(objs)
    result = pipeline.predict(user_query)
    return result

@app.get("/command")
def command(query: str, obj: str):
    result = scene_command(query, obj)
    return result

# uvicorn server:app --host 127.0.0.1 --port 8000