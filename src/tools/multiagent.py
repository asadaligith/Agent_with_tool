from agents import Agent , Runner, OpenAIChatCompletionsModel, RunConfig, set_tracing_disabled, function_tool, enable_verbose_stdout_logging, ModelSettings
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os

enable_verbose_stdout_logging()
load_dotenv()
set_tracing_disabled(True)
# '---------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable is not set.") 

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

configuration = RunConfig(OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client=external_client
))

@function_tool(is_enabled=False)
async def weather_tool(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."

@function_tool
async def calculator_tool(expression: str) -> str:
    """Calculate the result of a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}."
    except Exception as e:
        return f"Error in calculation: {e}"
    

translator_agent = Agent(
    name="translator agent",
    instructions="You are a translator agent you can translate text between different languages.",
    handoff_description="Translates the user query to the desired language if the query is related to translation.",
    )

orchestrator_agent = Agent(
    name="orchestrator agent",
    instructions="You are an orchestrator agent that decides which tool to use or call agent based on the user's query.",
    handoffs=[translator_agent],
    tools=[weather_tool, calculator_tool],
    model_settings=ModelSettings(tool_choice="required", max_tool_calls=2, temperature=0.7)
    )
query = input("Enter your query: ")
async def main():
    result = await Runner.run(orchestrator_agent, input=query, run_config=configuration)
    print("Final Output:", result.final_output)



def start():
   asyncio.run(main())