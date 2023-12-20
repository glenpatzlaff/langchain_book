import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, load_tools
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.utilities import OpenWeatherMapAPIWrapper

os.environ ["OPENAI_API_KEY"] = "<YOUR API KEY>"

# Initializing model with temperature
model = ChatOpenAI(temperature=0)

# Setting OpenWeatherMap API key as an environment variable
os.environ["OPENWEATHERMAP_API_KEY"] = "<YOUR API KEY>"

# Initializing OpenWeatherMapAPIWrapper
weather = OpenWeatherMapAPIWrapper()

# Loading tools and binding them with the model
tools = load_tools(["llm-math", "wikipedia", "openweathermap-api"], llm=model)
model_with_tools = model.bind(
    functions=[format_tool_to_openai_function(t) for t in tools]
)


# Chat prompt template with placeholders and system messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and knowledgeable assistant"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


# Defining the agent with scratchpad
agent = (
    {"input": lambda x: x["input"],
     "agent_scratchpad": lambda x: format_to_openai_functions(x["intermediate_steps"])}
    | prompt_template
    | model_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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
        st.chat_message("assistant").write(assistant_response["content"])
