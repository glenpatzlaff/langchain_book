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


review_prompt = PromptTemplate.from_template(
    """You are a critic. Given a story, it is your job to write an unbiased review for that story.

    Title: {title}
    Story: {story}"""
)


# Setting up chains for generating a title, story, and review using the model and StrOutputParser for output formatting
title_chain = title_prompt | model | StrOutputParser()
story_chain = story_prompt | model | StrOutputParser()
review_chain = review_prompt | model | StrOutputParser()
chain = (
       {"title": title_chain}
       | RunnablePassthrough.assign(story=story_chain)
       | RunnablePassthrough.assign(review=review_chain)
)


# Executing the chain if the user has entered a topic
if topic:
    # Display spinner while the content is being created
    with st.spinner("Creating story title, story and review"):
        # Invoking chain topic and storing the result
        result = chain.invoke({"topic": topic})
        # Displaying the generated results
        st.header(result["title"])
        st.write(result["story"])
        st.write(result["review"])
