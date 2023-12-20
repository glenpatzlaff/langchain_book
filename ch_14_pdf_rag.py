import os
import chainlit as cl
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig

os.environ ["OPENAI_API_KEY"] = "<YOUR API KEY>"

model = ChatOpenAI(streaming=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# Function triggered at the start of chat
@cl.on_chat_start
async def on_chat_start():
    # Request the user to upload a PDF file
    files = None
    while not files:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file",
            accept=["application/pdf"],
            max_size_mb=25,
            timeout=180,
        ).send()

    file = files[0]

    # Notifying the user about processing
    processing_msg = cl.Message(
        content=f"Processing `{file.name}`...",
        disable_human_feedback=True
    )
    await processing_msg.send()

    # Saving the uploaded file locally
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    with open(f"tmp/{file.name}", "wb") as f:
        f.write(file.content)

    # Loading and splitting the PDF document
    pdf_loader = PyPDFLoader(file_path=f"tmp/{file.name}")
    documents = pdf_loader.load_and_split(text_splitter=text_splitter)

    # Creating embeddings for the documents
    embeddings = OpenAIEmbeddings()
    doc_search = await cl.make_async(Chroma.from_documents)(documents, embeddings)

    # Inform the user about readiness
    processing_msg.content = f"I am ready to answer questions about `{file.name}`."
    await processing_msg.update()

    # Setting up the chat prompt template
    prompt_template = ChatPromptTemplate.from_template(
        """Answer questions based on the following: {context}

        Question: {question}"""
    )

    # Function to format document content
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Setting up the retriever and runnable
    retriever = doc_search.as_retriever()
    runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | model
            | StrOutputParser()
    )

    # Storing the runnable in the user session
    cl.user_session.set("runnable", runnable)


# Function triggered upon receiving a message
@cl.on_message
async def on_message(message: cl.Message):
    # Retrieve the runnable from the user session
    runnable = cl.user_session.get("runnable")  # type: Runnable

    # Create a message object for response
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Stream the response from the runnable
    async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await response_msg.stream_token(chunk)

    # Update the message with the final response
    await response_msg.update()
