from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from openai.types.shared import reasoning

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o-mini")
react_promp = hub.pull("hwchase17/react")

## reasoning agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_promp)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main():
    print("hello from Langchain-react-agent test")
    result = chain.invoke(
        input={
            "input": "Search for 3 SDET job openings using langchain in Hyderabad area with 12+ years of experience on linkedin and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
