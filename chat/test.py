import openai
import os

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7078"

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system",
         "content": "I am a normal housewife in Saudi Arabic"},
        {"role": "user",
         "content": """
           Tell me what is the first thing 
           you will do in the morning of a normal work day
        """},
    ]
)

print(response)
