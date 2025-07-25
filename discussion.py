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
    model1 = DiscussionModel("Analyst_A", "discussion_model_1_prompt.txt")
    model2 = DiscussionModel("Analyst_B", "discussion_model_2_prompt.txt")

    if os.path.exists("decisions.txt"):
        os.remove("decisions.txt")
    if os.path.exists("discussion.txt"):
        os.remove("discussion.txt")

    # Phase 1: Independent Proposals
    discussion_history = "Phase 1: Independent Proposals\n\n"

    result1 = model1.discuss(user_prompt, "")
    proposal1 = result1['suggestion']
    discussion_history += f"Analyst_A's Initial Proposal:\n{proposal1}\n\n"
    with open("discussion.txt", "w") as f:
        f.write(discussion_history)

    result2 = model2.discuss(user_prompt, "")
    proposal2 = result2['suggestion']
    discussion_history += f"Analyst_B's Initial Proposal:\n{proposal2}\n\n"
    with open("discussion.txt", "a") as f:
        f.write(f"Analyst_B's Initial Proposal:\n{proposal2}\n\n")

    # Phase 2: Discussion and Consensus
    discussion_history += "Phase 2: Discussion and Consensus\n\n"
    with open("discussion.txt", "a") as f:
        f.write("Phase 2: Discussion and Consensus\n\n")

    decisions = []
    while len(decisions) < 2:
        # Model 1
        result1 = model1.discuss(user_prompt, discussion_history)
        if result1["submitted"]:
            if "decision" in result1:
                decisions.append(result1["decision"])
            discussion_history += f"Analyst_A has submitted its decision.\n"
            with open("discussion.txt", "a") as f:
                f.write("Analyst_A has submitted its decision.\n")
        elif "suggestion" in result1 and result1["suggestion"] is not None:
            discussion_history += f"Analyst_A: {result1['suggestion']}\n\n"
            with open("discussion.txt", "a") as f:
                f.write(f"Analyst_A: {result1['suggestion']}\n\n")

        if len(decisions) == 2:
            break

        # Model 2
        result2 = model2.discuss(user_prompt, discussion_history)
        if result2["submitted"]:
            if "decision" in result2:
                decisions.append(result2["decision"])
            discussion_history += f"Analyst_B has submitted its decision.\n"
            with open("discussion.txt", "a") as f:
                f.write("Analyst_B has submitted its decision.\n")
        elif "suggestion" in result2 and result2["suggestion"] is not None:
            discussion_history += f"Analyst_B: {result2['suggestion']}\n\n"
            with open("discussion.txt", "a") as f:
                f.write(f"Analyst_B: {result2['suggestion']}\n\n")

    return discussion_history
