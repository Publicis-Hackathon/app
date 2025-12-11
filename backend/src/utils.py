from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


async def get_object_name(description: str) -> str:
    """Use LLM call to get a concise object name from descriptions."""
    llm = ChatOpenAI(model="gpt-4.1-nano")
    
    prompt_template = PromptTemplate(
        input_variables=["description"],
        template="Given the following object description: {description}, provide a concise name that best represents the element."
    )
    llm_chain = prompt_template | llm
    return await llm_chain.ainvoke(input={"description": description})