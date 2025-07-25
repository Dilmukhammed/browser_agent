import openai
import os
from dotenv import load_dotenv

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

    def coordinate(self, discussion_history, user_prompt):
        self.messages.append({"role": "user", "content": f"The user's request is: {user_prompt}\n\nHere is the discussion between the two models:\n{discussion_history}"})

        response = self.openai_client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=self.messages,
        )
        decision = response.choices[0].message.content
        return decision

def run_coordinator(user_prompt):
    with open("discussion.txt", "r") as f:
        discussion_history = f.read()

    coordinator = Coordinator("coordinator_model_prompt.txt")
    decision = coordinator.coordinate(discussion_history, user_prompt)

    print("--- Coordinator's Decision ---")
    print(decision)

    return decision
