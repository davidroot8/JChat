import ollama
print("this is the start")

ollama.pull("llama2")

reply = ollama.chat(
    model="llama2",
    messages=[{
        "role": "user",
        'content': 'sing me a happy birthday song'

    }]
)
print(reply)
print("hello world")