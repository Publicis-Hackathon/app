from src.tools.generate_image import generate_image
from src.logger import get_logger

logger = get_logger(__name__)

import os
from typing import Annotated, Dict

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import AgentState
from langgraph.graph.message import add_messages



DB_URI = os.getenv("CHECKPOINTS_URL", "file:src/checkpoints.db")


class State(AgentState):
    """
    This is the langgraph state for the filegpt agent.
    """
    # These two are mandatory for create_react_agent built in langgraph function
    messages: Annotated[list, add_messages]
    remaining_steps: int

    image_url: str
    structured_prompt: str
    segmentations: Dict
    

def run_agent(
    message: str,
    chat_id: str
) -> str:
    """Run the agent with the given message and chat ID."""
    model = ChatOpenAI(model="gpt-4.1-nano")
    prompt = """You are a helpful assistant. 
    Only generate one image at a time if asked. 
    The image is parsed and displayed separately. Do not response with the image URL directly."""
    agent = create_agent(
        model=model,
        tools=[generate_image],
        system_prompt=prompt,
        checkpointer=InMemorySaver(),  
        state_schema=State,
    )

    result = agent.invoke(
        input={
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        config={"configurable": {"thread_id": chat_id}}
    )
    content = result.get("messages", {})[-1].content
    
    output = {
        "content": content, 
        "image_url": result.get("image_url", None),
        "structured_prompt": result.get("structured_prompt", None),
    }
    return output