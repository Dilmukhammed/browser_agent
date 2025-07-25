import openai
import os
from dotenv import load_dotenv
import json

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
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_decision",
                    "description": "Submit your final decision to the coordinator.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": "Your final decision on the action to be taken."
                            }
                        },
                        "required": ["decision"]
                    }
                }
            }
        ]

    def discuss(self, user_prompt, discussion_history):
        self.messages.append({"role": "user", "content": user_prompt})
        if discussion_history:
            self.messages.append({"role": "system", "content": f"Here is the discussion so far:\n{discussion_history}"})

        response = self.openai_client.chat.completions.create(
            model="gemini-2.5-flash-lite-preview-06-17",
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
        )

        response_message = response.choices[0].message
        self.messages.append(response_message)

        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                if function_name == "submit_decision":
                    args = json.loads(tool_call.function.arguments)
                    decision = args.get("decision")
                    with open("decisions.txt", "a") as f:
                        f.write(f"{self.model_id}: {decision}\n")
                    return {"submitted": True, "decision": decision}

        suggestion = response_message.content
        self.messages.append({"role": "assistant", "content": suggestion})
        return {"submitted": False, "suggestion": suggestion}

def run_discussion(user_prompt):
    model1 = DiscussionModel("cautious_model", "discussion_model_1_prompt.txt")
    model2 = DiscussionModel("adventurous_model", "discussion_model_2_prompt.txt")

    discussion_history = ""
    decisions = []

    if os.path.exists("decisions.txt"):
        os.remove("decisions.txt")

    while len(decisions) < 2:
        # Model 1
        result1 = model1.discuss(user_prompt, discussion_history)
        if result1["submitted"]:
            decisions.append(result1["decision"])
            discussion_history += f"Cautious Model has submitted its decision. Adventurous Model, please submit your decision.\n"
            with open("discussion.txt", "w") as f:
                f.write(discussion_history)
        else:
            discussion_history += f"Cautious Model: {result1['suggestion']}\n\n"
            with open("discussion.txt", "w") as f:
                f.write(discussion_history)

        if len(decisions) == 2:
            break

        # Model 2
        result2 = model2.discuss(user_prompt, discussion_history)
        if result2["submitted"]:
            decisions.append(result2["decision"])
            discussion_history += f"Adventurous Model has submitted its decision. Cautious Model, please submit your decision.\n"
            with open("discussion.txt", "a") as f:
                f.write(discussion_history)
        else:
            discussion_history += f"Adventurous Model: {result2['suggestion']}\n\n"
            with open("discussion.txt", "a") as f:
                f.write(f"Adventurous Model: {result2['suggestion']}\n\n")

    return discussion_history
