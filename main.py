from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv(override=True)

core_version=version("langchain_core")
lg_version=version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

print(f"LangChain Core Version: {core_version}")
print(f"LangGraph Version: {lg_version}")

def main():
    # test openai
    llm=ChatOpenAI(model="gpt-5-nano", temperature=0)
    response = llm.invoke("say! 'setup completed' ! in one word")
    print(response.content)

    #test anthropic 
    llm_anthropic = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    response = llm_anthropic.invoke("say! 'setup completed' ! in one word")
    print(response.content)


if __name__ == "__main__":
    main()
