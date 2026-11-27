"""
langchain core concepts - LCEL and Runnables
"""

from docstring_parser import parser
from anthropic.types import model
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

def demo_basic_chain():
    """ Demopnstrate a basic chain using LCEL AND Runnables"""

    #component 1 : define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpfull assistance
        Answer in one sentence:{question}"""
    )
    #component 2: define the LLM
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)

    #component 3: define the output parser
    parser = StrOutputParser()

    #component 4: assemble the chain using LCEL
    chain = prompt| model| parser

    #component 5: invoke the chain
    response= chain.invoke({"question":"What is the capital of France?"})
    print(response)

    return chain

def demo_batch_executions():
    """ Demonstrate batch execution of a chain """

    prompt = ChatPromptTemplate.from_template("translate to french:{text}")
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt| model| parser

    #batch inputs
    inputs=[
        {"text": "Hello"},
        {"text": "How are you?"},
        {"text": "What is your name?"}
    ]
    
    results = chain.batch(inputs)
    print(results)

    for text in zip(inputs,results):
        print(f"Input:{text[0]["text"]} ==> output:{text[1]}")

    return chain


def demo_streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template(
        "write a haiku about :{topic}"
    )
    model = ChatOpenAI(model="gpt-5-nano", temperature=0.7)
    parser = StrOutputParser()
    chain = prompt| model| parser

    #stream output
    print("\n--- Stream Output ---")
    for chunk in chain.stream({"topic":"blue stars"}):
        print(chunk, end="", flush=True)
    print()# for new line after streaming

def demo_schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")

# ------- Exercise the demos -------#
# Exercise: Build your first chain
def exercise_first_chain():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """
    prompt = ChatPromptTemplate.from_template(
        "create a marketing tag line for a product name '{product}'targeting '{audience}'."
    )
    model=ChatOpenAI(
        model="gpt-5-nano",
        temperature=0.7
    )
    parser = StrOutputParser()

    #Assemble the chain
    chain = prompt | model | parser

    response = chain.invoke({"product":"AI Course","audience":"developers"})
    print(response)

def new_way():
    # the univeral way to initialize a model
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=1500)

    # Or provider-specific (still works)

    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic

    openai_model = ChatOpenAI(model="gpt-4o-mini",
                              temperature=0.7,
                              max_tokens=1500,
                              timeout=30,
                              max_retries=3)
    
    anthropic_model = ChatAnthropic(model="claude-sonnet-4-5-20250929")



if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_executions()
    # demo_streaming()
    # demo_schema_inspection()
    exercise_first_chain()
