import openai
import os
from dotenv import load_dotenv
import time

load_dotenv()

class Coordinator:
    def __init__(self, prompt_file):
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("GEMINI_TOKEN"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def coordinate(self, decisions, user_prompt):
        self.messages.append({"role": "user", "content": f"The user's request is: {user_prompt}\n\nHere are the decisions from the two models:\n{decisions}"})

        response = self.openai_client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=self.messages,
        )
        decision = response.choices[0].message.content
        return decision

def run_coordinator(user_prompt):
    while not os.path.exists("decisions.txt"):
        time.sleep(1)

    while True:
        with open("decisions.txt", "r") as f:
            decisions = f.readlines()
        if len(decisions) >= 2:
            break
        time.sleep(1)

    decisions_str = "".join(decisions)
    coordinator = Coordinator("coordinator_model_prompt.txt")
    decision = coordinator.coordinate(decisions_str, user_prompt)

    print("--- Coordinator's Decision ---")
    print(decision)

    return decision
