import asyncio
from discussion import run_discussion
from coordinator import run_coordinator
from worker import Worker

async def main():
    # Get user input
    user_prompt = "Find the price of the cheapest laptop on amazon.com"

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
        await worker.run(decision)
    finally:
        await worker.close()

if __name__ == "__main__":
    asyncio.run(main())
