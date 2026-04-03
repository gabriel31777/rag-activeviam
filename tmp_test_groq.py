import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")
print(f"Key found: {key[:10]}...")

client = Groq(api_key=key)

completion = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": "Hello, answer in 1 word."}
    ]
)

print(f"Response: {completion.choices[0].message.content}")
