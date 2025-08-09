from agents import Agent , Runner , OpenAIChatCompletionsModel , set_tracing_disabled , function_tool
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import chainlit as cl

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
set_tracing_disabled(True)

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"   
)

Model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

@function_tool
async def get_weather(city:str) -> str:
    """
    Provide the current weather for the given city.
    
    """
    return f"The current weather in {city} is sunny."


weather_agent = Agent(
    name="Weather Agent",
    instructions=
    """
    you are an expert in weather dignoses, and good knowlede any other relted informations, you give answer user quries,
      if the user asq about weather then using tool.
    """,
    tools=[get_weather],
    model=Model
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome to the Weather Agent! Ask me anything about the weather.").send()

@cl.on_message
async def on_message(message: cl.Message):
    response = await Runner.run(
        weather_agent,
        input=message.content,
    )
    await cl.Message(content=response.final_output).send()
