from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool


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

prompt = ChatMessagePromptTemplate.format_messages(
  [
    (
      "system",
      """
      You are a research assisstant that will help generate a research paper.
      Answer the user query and use necessary tools.
      Wrap the output in this format and provide no other text\n{format_instructions}
      """,
      
    ),
    ("paceholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
  ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool]
agent = create_tool_calling_agent(
  llm=llm,
  prompt=prompt,
  tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What do you want to search? ")
raw_response = agent_executor.invoke({"query": "What is 7th wonder?"})
print(raw_response)

structured_response = parser.parse(raw_response.get("output")[0]["text"])
print(structured_response.topic)

try: 
  structured_response = parser.parse(raw_response.get("output")[0]["text"])
  print(structured_response)
except Exception as e:
  print("Error parsing response", e, "Raw Response -", raw_response)