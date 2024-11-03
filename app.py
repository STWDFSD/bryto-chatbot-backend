import os
import time
import boto3
import pandas as pd
from io import StringIO
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from mangum import Mangum

from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE")
OPENAI_MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_response(prefix: str, message: str):
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_MODEL_NAME,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    pc = Pinecone(api_key=PINECONE_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index = pc.describe_index(PINECONE_INDEX)
    vectorstore = PineconeVectorStore(pinecone_api_key=PINECONE_KEY, index_name=PINECONE_INDEX, embedding=embeddings, namespace=PINECONE_NAMESPACE)
    retriever = vectorstore.as_retriever()

    SYSTEM_TEMPLATE = "Answer the user's questions based on the below context. " + prefix + """ 
        Answer based on the only given theme. 
        Start a natural-seeming conversation about anything that relates to the lesson's content.

        <context>
        {context}
        </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant "
                "to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )
    stream = conversational_retrieval_chain.stream(
        {
            "messages": [
                HumanMessage(content=message),
            ],
        }
    )

    async def event_generator():
        all_content = ""
        for chunk in stream:
            for key in chunk:
                if key == "answer":
                    all_content += chunk[key]
                    yield f'data: {chunk[key]}\n\n'

    return event_generator()


def read_csv_from_s3(bucket_name, file_key):
    s3 = boto3.client('s3', 
                      aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY,
                      region_name='eu-north-1')
    
    # Get the object from S3
    print(f"Attempting to read file from bucket: {bucket_name}, with key: {file_key}")
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    
    print(obj)
    # Read the object's content into a pandas DataFrame
    csv_data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
    
    return csv_data

class Document:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata if metadata is not None else {}


@app.get("/chat")
async def sse_request(prefix: str = '', message: str = ''):
    return StreamingResponse(get_response(prefix, message), media_type='text/event-stream')

@app.get("/train")
def train(bucket_name: str, file_key: str):
    try:
        # bucket_name = 'homebuyer-llm-datasets'
        # file_key = 'community_data/community_info.csv'
        # Read CSV data from S3
        csv_data_frame = read_csv_from_s3(bucket_name, file_key)

        # Convert DataFrame to list of Document objects with metadata
        documents = [
            Document(
                content=row.to_json(),
                metadata={'row_index': index}
            ) 
            for index, row in csv_data_frame.iterrows()
        ]

        # Initialize OpenAI Embeddings
        embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_KEY)

        index_name = PINECONE_INDEX
        namespace = PINECONE_NAMESPACE

        # Check if the index already exists and has the correct dimension
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        # Create or reconfigure the index with the correct dimensions
        if index_name in existing_indexes:
            print(f"Index {index_name} already exists. Deleting existing index to recreate with correct dimensions.")
            pc.delete_index(index_name)
            time.sleep(30)  # Ensure the index is fully deleted

        # Create the index with the desired dimension (1536)
        pc.create_index(
            name=index_name,
            dimension=1536,  # Ensure this matches the dimension of your embeddings
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

        # Wait until the index is ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

        # Get the index instance
        index = pc.Index(index_name)

        # Create a PineconeVectorStore from the documents
        docsearch = PineconeVectorStore.from_documents(
            documents,
            embeddings_model,
            index_name=index_name,
            namespace=namespace,
        )

    except Exception as e:
        return {"status": "Error", "message": str(e)}


# @app.post("/feedback")
# async def feedback(request: Request):
#     body = await request.json()

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a helpful assistant. Answer to the best of your ability.",
#             ),
#             MessagesPlaceholder(variable_name="chat_history"),
#             (
#                 "user",
#                 "Analyzes the following conversation and provides possible feedback, such as grammar or "
#                 "spelling errors, all about the HumanMessage. Respond using markdown."
#             ),
#         ]
#     )

#     chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL_NAME)

#     chat_history = ChatMessageHistory()
#     for item in body["chat_history"]:
#         if item["who"] == "ai":
#             chat_history.add_ai_message(item["text"])
#         else:
#             chat_history.add_user_message(item["text"])

#     chain = prompt | chat

#     response = chain.invoke(
#         {
#             "chat_history": chat_history.messages,
#         }
#     )

#     return {"content": response.content}

@app.get("/")
async def hello_world():
    return {"status": "Server is running..."}

# AWS Lambda handler
handler = Mangum(app)