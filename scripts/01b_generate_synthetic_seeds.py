import asyncio
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Ensure we can import src
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jugaad_bench.models import RawCase

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("synthetic_seeds")

load_dotenv()

class RawCaseList(BaseModel):
    cases: list[RawCase] = Field(..., description="List of at least 50 distinct grassroots innovation cases.")

async def generate_seeds():
    api_key = os.environ.get("OPENAI_API_KEY")
    client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

    prompt = """
    Generate exactly 50 HIGHLY DIVERSE, REALISTIC grassroots engineering innovation cases from rural India, Africa, South America, and Southeast Asia.
    These must be "Jugaad" or frugal innovations—creative workarounds using cheap, locally available, or repurposed materials to solve physical problems under resource constraints.
    Examples: bamboo windmills, motorcycle-powered water pumps, clay refrigerators, bicycle-powered washing machines, plastic bottle solar bulbs.
    
    For each case, provide:
    - source_id: A unique ID (e.g., SYN-001, SYN-002)
    - source_url: A hypothetical descriptive URL (e.g., https://example.com/bamboo-windmill)
    - title: Clear title of the innovation
    - raw_text: A detailed 2-3 paragraph description of the problem context, the innovator, the materials used, and the underlying physical mechanism of the solution.
    
    Ensure high variety in domains: agriculture, water, energy, household, health, transport.
    """

    logger.info("Calling GPT-4o to synthesize 50 seeds...")
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            response_model=RawCaseList,
            messages=[
                {"role": "system", "content": "You are a mechanical engineering historian specializing in grassroots frugal innovations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=16384,
            timeout=120
        )
        
        out_path = Path("data/raw/all_raw_cases.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.model_dump() for c in response.cases], f, indent=2)
            
        logger.info(f"Successfully generated {len(response.cases)} seeds to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(generate_seeds())
