from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI 

from tavily import TavilyClient


load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that search on Internet
    Args:
        query: The query for Search
    Return:
        The search result
    """
    print(f"Seaching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("this is tavily usage example")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 SDET job openings in Hyderabad area with 12+ years of experience")})
    print(result)

if __name__=="__main__":
    main()

