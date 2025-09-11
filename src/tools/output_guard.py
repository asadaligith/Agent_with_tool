from agents import Agent , Runner , OpenAIChatCompletionsModel , RunConfig , RunContextWrapper , set_tracing_disabled, output_guardrail , GuardrailFunctionOutput, OutputGuardrailTripwireTriggered
from openai import AsyncOpenAI
from dotenv import load_dotenv
import asyncio
import os
from pydantic import BaseModel , Field


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

class check_response(BaseModel):
    output_info:str 
    is_english_translation_related:bool = Field(discription="Indicates if the ouput is related to translation into english must retrun true.") 

english_guard_agent = Agent(
    name="english guard agent",
    instructions="You are an english guard agent that you evil eye on final output, its not allow any other output except translate into english.",
    output_type=check_response,
)

@output_guardrail
async def english_translation_guardrail(ctx:RunContextWrapper, agent:Agent, output:str) -> GuardrailFunctionOutput:
    result = await Runner.run(english_guard_agent, input=output, run_config=configuration)
    return GuardrailFunctionOutput(
        output_info=result.final_output.output_info,
        tripwire_triggered= not result.final_output.is_english_translation_related,
    )

english_translator_agent = Agent(
    name="english translator agent",
    instructions="You are an english translator agent that translates user queries to english.",
    handoff_description="Translates the user query to english if the query is related to translate into english.",
    output_guardrails=[english_translation_guardrail]
)


agent = Agent(
    name="helpful agent",
    instructions="You are a helpful agent that assists users with their queries, if the query related to translate into english, call to english translator agent.",
    handoffs=[english_translator_agent],
    )

query = input("Enter your query: ")
async def main():

    try:
        result = await Runner.run(agent, input=query, run_config=configuration)
        print("Final Output:", result.final_output)

    except OutputGuardrailTripwireTriggered as e:
        print("❌Output Guardrail Triggered:", e)
def start():
    asyncio.run(main())