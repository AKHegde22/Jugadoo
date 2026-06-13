import asyncio
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def test():
    models = await client.models.list()
    names = [m.id for m in models.data if "gpt" in m.id or "o1" in m.id or "o3" in m.id]
    print(names)

asyncio.run(test())
