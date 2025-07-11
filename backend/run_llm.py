import os
from langchain_google_vertexai import ChatVertexAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import history_aware_retriever
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever


def run_llm(query, chat_history):

    llm = ChatVertexAI(model_name="gemini-2.5-flash")
    client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
    vector_store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME_LANGCHAIN"], embedding=embeddings
    )

    retrieval_prompt_template = """
    You are a documentation helper. Be practical and always try your best to help the user about their questions based on the documentation.
    
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>

    User's question: {input}
    """

    retrieval_prompt = PromptTemplate.from_template(template=retrieval_prompt_template)

    rephrase_prompt = client.pull_prompt("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(search_kwargs={"k": 12}), rephrase_prompt
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt=retrieval_prompt)

    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    res = retrieval_chain.invoke({"input": query, "chat_history": chat_history})
    return res


if __name__ == "__main__":
    while True:   
        user_input = input("input: ")
        if(user_input == "/bye"):
            break

        hist = [{"role":"user","content":"what is langchain"},{"role":"assistant","content":"Based solely on the provided context, there is no explicit definition of what \"LangChain\" is. The context primarily details different modules, classes, and functionalities available within the LangChain Python API, such as \"Chains\" and \"LangSmith utilities.\" \n\n --Resources-- \n\n\n\n https://python.langchain.com/v0.2/api_reference/nomic/index.html\n\nhttps://python.langchain.com/api_reference/langchain/chains.html\n\nhttps://python.langchain.com/v0.2/api_reference/langchain/chains.html\n\nhttps://python.langchain.com/api_reference/cohere/index.html\n\nhttps://python.langchain.com/api_reference/snowflake/index.html\n\nhttps://python.langchain.com/v0.2/api_reference/community/chains.html\n\nhttps://python.langchain.com/v0.2/api_reference/_modules/index.html\n\nhttps://python.langchain.com/v0.2/api_reference/airbyte/index.html\n\nhttps://python.langchain.com/v0.2/api_reference/ai21/llms.html\n\nhttps://python.langchain.com/api_reference/community/chains.html\n\nhttps://python.langchain.com/api_reference/langchain/smith.html\n\nhttps://python.langchain.com/v0.2/api_reference/langchain/smith.html"}]
        res = run_llm(user_input, hist)
        print(res["answer"])
