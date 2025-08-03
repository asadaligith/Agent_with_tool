from agents import Agent , Runner , OpenAIChatCompletionsModel , set_tracing_disabled 
from openai import AsyncOpenAI 
from dotenv import load_dotenv
import os

load_dotenv()
set_tracing_disabled(True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

Model = OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client=external_client
)

Frontend_agent = Agent(
    name="Frontend Agent",
    instructions=
    """
    You are expert in Frontend you give answer to user about frontend related query
    """,
    model=Model
)


def main():   
    response = Runner.run_sync(
        Frontend_agent,
        input="what is Frontend",
    )
    print(response.final_output)
