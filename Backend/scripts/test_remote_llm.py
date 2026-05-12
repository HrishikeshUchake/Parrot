import asyncio
import logging
import traceback
from Backend.llm.llm_provider import get_llm_provider_for_backend

logging.basicConfig(level=logging.DEBUG)

async def main():
    print("Initializing remote LLM client...")
    try:
        client = get_llm_provider_for_backend("openai")
        
        prompt = "Explain in one short sentence what a parrot is."
        print(f"Sending prompt: {prompt}")
        
        response = await client.generate(prompt)
        print("\n=== Success ===")
        print(f"Response: {response}")
        
    except Exception as e:
        print("\n=== Failed ===")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
