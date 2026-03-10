from command_parser import parse_command
from semantic_search import search_object
from fastapi import FastAPI

app = FastAPI()

def scene_command(user_query, objects):

    detected_object = search_object(user_query, objects)
    command = parse_command(user_query, detected_object)
    return command

@app.get("/command")
def command(query: str, obj: str):
    result = scene_command(query, obj)
    return result

# uvicorn main:app --host 127.0.0.1 --port 8000