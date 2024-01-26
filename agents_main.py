from langchain.agents import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import pymysql

pymysql.install_as_MySQLdb()

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv('openai_api_key')
conn_string = "sqlite:///Chinook.db"


def get_response(question):
    # with open('./prompts/sql_query_generation_prompt.txt', 'r', encoding='utf-8') as file:
    #     prompt_template = file.read()

    db = SQLDatabase.from_uri(conn_string)

    agent_executor = create_sql_agent(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key,
                       model="gpt-3.5-turbo"),
        toolkit=SQLDatabaseToolkit(db=db, llm=ChatOpenAI(temperature=0,
                                                         openai_api_key=openai_api_key,
                                                         model="gpt-3.5-turbo")),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
        max_iterations=20
    )

    ai_response = agent_executor.invoke(question)

    return ai_response['output']


if __name__ == '__main__':
    response = get_response('Find all albums for the artist AC/DC')
    print(response)
