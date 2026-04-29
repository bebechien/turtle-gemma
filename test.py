from config import MODEL_ID
from turtle_engine import HeadlessTurtle
from gemma_agent import GemmaAgent

turtle_engine = HeadlessTurtle()

ai_agent = GemmaAgent(MODEL_ID)
result = ai_agent.run_interactive("draw a green star", turtle_engine, 1)

while True:
    try:
        chunk = next(result)
        print(chunk, end="", flush=True)
    except StopIteration as e:
        break

print("- End of Test -")
print(f"{turtle_engine.get_history_text()}")

