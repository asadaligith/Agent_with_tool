from agents import Agent , Runner , OpenAIChatCompletionsModel , RunConfig , set_tracing_disabled, input_guardrail , RunContextWrapper , GuardrailFunctionOutput
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel , Field
import asyncio
import os

# .---------------------------------------------------------------
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

configioration = RunConfig(OpenAIChatCompletionsModel(
    model= "gemini-2.5-flash",
    openai_client=external_client
))

class Output_Check(BaseModel):
    output_info:str
    is_prime_minister_related:bool = Field(discription="Indicates if the query is related to the prime minister must retrun true.")

guard_Agent = Agent(
    name="guard agent",
    instructions="You are a guard agent that you evil eye on user queries,  its should be not allow any queries related to prime minister.",
    output_type=Output_Check
)

@input_guardrail
async def prime_minister_guardrail(ctx :RunContextWrapper, agent :Agent ,input: str )-> GuardrailFunctionOutput:
    result =await Runner.run(guard_Agent, input=input, run_config=configioration)
    
    return GuardrailFunctionOutput(
        output_info=result.final_output.output_info,
        tripwire_triggered=result.final_output.is_prime_minister_related,
    )

agent = Agent(
    name="helpful agent",
    instructions="You are a helpful agent that assists users with their queries.",
    input_guardrails=[prime_minister_guardrail],
    )

query = input("Enter your query: ")
async def main():

    try:
        result = await Runner.run(agent, input=query, run_config=configioration)
        print("Final Output:", result.final_output)

    except Exception as e:
        print("Error:", (e))

def start():
    asyncio.run(main())