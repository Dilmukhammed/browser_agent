import openai
import os
from dotenv import load_dotenv

load_dotenv()

class DiscussionModel:
    def __init__(self, model_id, prompt_file):
        self.model_id = model_id
        with open(prompt_file, 'r') as f:
            self.system_prompt = f.read()
        self.openai_client = openai.OpenAI(
            api_key=os.getenv("GEMINI_TOKEN"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def discuss(self, user_prompt, discussion_history):
        self.messages.append({"role": "user", "content": user_prompt})
        if discussion_history:
            self.messages.append({"role": "system", "content": f"Here is the discussion so far:\n{discussion_history}"})

        response = self.openai_client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=self.messages,
        )
        suggestion = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": suggestion})
        return suggestion

def run_discussion(user_prompt):
    model1 = DiscussionModel("cautious_model", "discussion_model_1_prompt.txt")
    model2 = DiscussionModel("adventurous_model", "discussion_model_2_prompt.txt")

    discussion_history = ""

    # Round 1
    suggestion1 = model1.discuss(user_prompt, discussion_history)
    discussion_history += f"Cautious Model: {suggestion1}\n\n"
    with open("discussion.txt", "w") as f:
        f.write(discussion_history)

    suggestion2 = model2.discuss(user_prompt, discussion_history)
    discussion_history += f"Adventurous Model: {suggestion2}\n\n"
    with open("discussion.txt", "a") as f:
        f.write(f"Adventurous Model: {suggestion2}\n\n")

    # Round 2
    suggestion1 = model1.discuss(user_prompt, discussion_history)
    discussion_history += f"Cautious Model: {suggestion1}\n\n"
    with open("discussion.txt", "a") as f:
        f.write(f"Cautious Model: {suggestion1}\n\n")

    suggestion2 = model2.discuss(user_prompt, discussion_history)
    discussion_history += f"Adventurous Model: {suggestion2}\n\n"
    with open("discussion.txt", "a") as f:
        f.write(f"Adventurous Model: {suggestion2}\n\n")

    return discussion_history
