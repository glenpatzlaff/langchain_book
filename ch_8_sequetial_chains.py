import os
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

os.environ ["OPENAI_API_KEY"] = "<YOUR API KEY>"

# Initializing the OpenAI model
model = OpenAI()

# Setting up a Streamlit web app title
st.title("Story Creator")
# Text input field for the user to enter a topic
topic = st.text_input("Choose a topic to create a story about")

title_prompt = PromptTemplate.from_template(
    "Write a great title for a story about {topic}"
)


story_prompt = PromptTemplate.from_template(
    """You are a talented writer. Given the title of a story, it is your job to write a story for that title.

    Title: {title}"""
)

# Setting up chains for title and story using the model and StrOutputParser for output formatting
title_chain = title_prompt | model | StrOutputParser()
story_chain = story_prompt | model | StrOutputParser()
chain = (
    {"title": title_chain}
    | RunnablePassthrough.assign(story=story_chain)
)

# Executing the chain if the user has entered a topic
if topic:
    # Displaying a spinner while content created
    with st.spinner("Creating story title and story"):
        # Invoking the chain, storing the result
        result = chain.invoke({"topic": topic})
        # Displaying the generated results
        st.header(result["title"])
        st.write(result["story"])
