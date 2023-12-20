import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferMemory
import requests
from langchain.agents import Tool

os.environ ["OPENAI_API_KEY"] = "<YOUR API KEY>"

# Initializing model with temperature
model = ChatOpenAI(model="gpt-4", temperature=0)

st.title("🍽️ Chefbot")


# Function to make API calls to TheMealDB
def make_api_call(base_url, params):
    """
    Makes an API call to TheMealDB and returns the JSON response.
    Args:
    base_url (str): The base URL for the API endpoint.
    params (dict): The parameters for the API request.


    Returns:
    dict: The JSON response from the API call.
    """
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Request failed with status code: {response.status_code}")
    return None


# Get recipes by category, ingredient, and ID
def get_recipes_by_category(category):
    return make_api_call(
        "https://www.themealdb.com/api/json/v1/1/filter.php", {"c": category})


def get_recipes_by_ingredient(ingredient):
    return make_api_call(
        "https://www.themealdb.com/api/json/v1/1/filter.php", {"i": ingredient})


def get_recipe_by_id(id):
    return make_api_call(
        "https://www.themealdb.com/api/json/v1/1/lookup.php", {"i": id})


# Define tools for working with TheMealDB API
tools = [
    Tool(name="RecipesByIngredient", func=get_recipes_by_ingredient,
         description="Useful for getting recipes based on an ingredient"),
    Tool(name="RecipesByCategory", func=get_recipes_by_category,
         description="Useful for getting recipes based on a category"),
    Tool(name="RecipeById", func=get_recipe_by_id,
         description="Useful for getting a specific recipe based on a recipe ID")
]

# Binding the tools with the ChatOpenAI model
model_with_tools = model.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)

# Setting prompt template with placeholders and system messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a sassy chef's assistant"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Defining the agent and memory
agent = (
        {"input": lambda x: x["input"],
         "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"]),
         "chat_history": lambda x: x["chat_history"]}
        | prompt_template
        | model_with_tools
        | OpenAIFunctionsAgentOutputParser()
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

# Streamlit UI setup
st.title("Assistant with Tools")

# Initialize or retrieve session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What can I help you with?"}]

# Display chat history in Streamlit UI
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Initialize or retrieve agent_executor session state
if "agent_executor" not in st.session_state:
    st.session_state["agent_executor"] = agent_executor

# Handle user input in Streamlit chat widget
if user_input := st.chat_input():
    with st.spinner("Working on it ..."):
        # Process and display user input
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Invoke agent_executor with the user's input to get the response
        agent_executor = st.session_state.agent_executor
        response = agent_executor.invoke({"input": user_input})["output"]

        # Prepare and display the assistant's response
        assistant_response = {"role": "assistant", "content": response}
        st.session_state.messages.append(assistant_response)
        st.chat_message("assistant").write(assistant_response["content"])

        # Update the agent_executor in the session state
        st.session_state.agent_executor = agent_executor
