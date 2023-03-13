import openai

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
