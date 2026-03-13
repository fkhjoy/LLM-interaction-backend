import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def parse_command(user_query, detected_object):

    prompt = f"""
        Allowed actions:
        focus
        highlight
        highlight_and_navigate
        move_camera
        inspect

        Return ONLY JSON with action, obj fields.

        User command: {user_query}
        Detected object: {detected_object}
        
        You convert user commands into JSON instructions.
        
        JSON:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.2
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    json_start = text.find("{")
    json_end = text.rfind("}") +1
    json_text = text[json_start:json_end]
    return json.loads(json_text)

if __name__ == "__main__":
    result = parse_command(
        "show me the entrance",
        "front_door"
    )

    print(result)