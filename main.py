import asyncio
from discussion import run_discussion
from coordinator import run_coordinator
from worker import Worker
import os

async def main():
    user_prompt = "Find the price of the cheapest laptop on amazon.com"
    action_history = []

    for i in range(3): # Loop for a few turns for demonstration
        print(f"--- Turn {i+1} ---")

        # Prepare the discussion context
        with open("discussion.txt", "w") as f:
            f.write(f"User Goal: {user_prompt}\n")
            f.write("Action History:\n")
            if not action_history:
                f.write("No actions taken yet.\n")
            else:
                for action in action_history:
                    f.write(f"- {action}\n")
            f.write("\n")

        # Run the discussion
        print("--- Running Discussion ---")
        run_discussion(user_prompt)

        # Run the coordinator
        print("\n--- Running Coordinator ---")
        decision = run_coordinator(user_prompt)

        # Run the worker
        print("\n--- Running Worker ---")
        worker = Worker()
        try:
            executed_action = await worker.run(decision)
            if executed_action:
                action_history.append(executed_action)
        finally:
            await worker.close()

if __name__ == "__main__":
    asyncio.run(main())
