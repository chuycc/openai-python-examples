import os
from openai import OpenAI

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are Chewbacca"},
        {
            "role": "user",
            "content": "How are you Chewie?",
        },
    ],
)

print(completion.choices[0].message.content)
