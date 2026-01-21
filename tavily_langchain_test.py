from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI 
from langchain_tavily import TavilySearch


load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("this is tavily usage example")
    result = agent.invoke({"messages":HumanMessage(content="Search for 3 SDET job openings in Hyderabad area with 12+ years of experience")})
    print(result)

if __name__=="__main__":
    main()

