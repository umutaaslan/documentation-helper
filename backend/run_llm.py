import os
from langchain_google_vertexai import ChatVertexAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langsmith import Client
from langchain.prompts import PromptTemplate

def run_llm(query):

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

    # retrieval_qa_prompt = client.pull_prompt("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt=retrieval_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(search_kwargs={"k": 12}), combine_docs_chain=combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": query})
    return res


if __name__ == "__main__":
    res = run_llm("what are the ways of memory implementation in langchain")
    print(res["answer"])
