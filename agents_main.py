import ast
import re
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_types import AgentType
import pymysql
pymysql.install_as_MySQLdb()

load_dotenv()
openai_api_key = os.getenv('openai_api_key')
os.environ['OPENAI_API_KEY'] = openai_api_key
conn_string = "sqlite:///Chinook.db"



def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))


def get_retreival_tool(db):
    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
    valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )

    return retriever_tool


def get_response(question):

    with open('prompt.txt', 'r', encoding='utf-8') as file:
        prompt = file.read()

    db = SQLDatabase.from_uri(conn_string)

    retriever_tool = get_retreival_tool(db=db)
    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key,
                       model="gpt-4"),
        toolkit=SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0,
                                                         openai_api_key=openai_api_key,
                                                         model="gpt-3.5-turbo")),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        suffix=prompt,
        extra_tools=[retriever_tool],
        handle_parsing_errors=True,
        max_iterations=20,
    )

    ai_response = agent_executor.invoke(question)

    return ai_response['output']



