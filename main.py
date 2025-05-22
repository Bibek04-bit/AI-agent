from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()
#can use claude or gpt
class ResearchResponse(BaseModel):
  topic: str
  summary: str
  sources: list[str]
  tools_used: list[str]


llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
# response = llm.invoke("Who is the father of Ai?")
# print(response)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
