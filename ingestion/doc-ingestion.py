import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


def ingest_docs(path: str, index_name: str):
    loader = ReadTheDocsLoader(path=path)
    raw_documents = loader.load()
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(documents=raw_documents)

    for chunk in chunks:
        curr_url = chunk.metadata["source"]
        keyword = "langchain-docs-latest"
        pos = curr_url.find(keyword)
        if pos != -1:
            new_url = "https:/" + curr_url[pos+len(keyword):]
            new_url = new_url.replace("langchain-docs-latest", "https:/")
            chunk.metadata.update({"source": new_url})
        else:
            print(f"{keyword} couldnt found to change metadata source")

    
    print(f"{len(chunks)} chunks will be upserted")

    def chunks_generator(chunks, size):
        for i in range(0, len(chunks), size):
            yield chunks[i: i+size]

    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    for chunk in chunks_generator(chunks=chunks, size=200):
        print(f"{len(chunk)} chunk is being upserted to vector store", end=" ")
        PineconeVectorStore.from_documents(documents=chunk, embedding=embeddings, index_name=index_name)
        print("\N{CHECK MARK}")

    print("done!")



if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "langchain-docs-latest")
    ingest_docs(path=path, index_name="langchain-docs-index-with-https-sources")