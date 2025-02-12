import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from datetime import datetime, timezone
from retriever import retrive_url

load_dotenv()

llm = AzureChatOpenAI(
    openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY")
)


urls = [
  "https://en.wikipedia.org/wiki/Viscosity"
]


url_retrieval_tool = retrive_url(urls)


tools = [url_retrieval_tool]

SYSTEM_PROMPT = """You are an AI assistant that specializes in answering questions using a retrieval tool.
You can call the following tool:
- retrieve_blog_posts: Search and return information about viscosity.
"""

system_message = SystemMessage(content=SYSTEM_PROMPT)
agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)

