import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory #allows to keep track of previous messages in the conversation
from langchain.chains import ConversationalRetrievalChain # allows to chat with the context in the vectorstore
from htmlTemplate import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="PDFPal", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question=st.text_input("Ask a question based on the uploaded documents: ")

    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Uploaded Documents")
        pdf_docs=st.file_uploader("Upload the PDFs here and click on Submit",
                         accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text=get_pdf_text(pdf_docs)
                # print(raw_text)
                # st.write(raw_text)

                #get text chunks
                text_chunks=get_text_chunks(raw_text)

                #create vector store
                vectorstore=get_vectorstore(text_chunks)

                #create a conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore) #allows to retrieve the conversation 
                #st.session_state indicates that this element won't be reinitialized during each session (not applicable to reloading the page)

if __name__=='__main__':
    main()