from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI 
from langchain_tavily import TavilySearch

from typing import List
from pydantic import BaseModel, Field


load_dotenv()

class Source(BaseModel):
    """Schema for source used by agent"""
    url:str = Field(description="this is the URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and source"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of soucres used by agent for response")

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():
    print("this is tavily usage example")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 SDET job openings in Hyderabad area with 12+ years of experience")})
    print(result)

if __name__=="__main__":
    main()