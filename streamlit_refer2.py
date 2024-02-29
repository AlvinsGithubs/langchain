import streamlit as st
import tiktoken
from loguru import logger
import gdown  # 구글 드라이브 다운로드를 위해 추가

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# OpenAI API 키를 여기에 입력하세요 (보안을 위해 환경 변수나 설정 파일 사용을 권장)
OPENAI_API_KEY = "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"

def main():
    st.set_page_config(page_title="GA_Chat", page_icon=":books:")
    st.title("_NSUSLAB :red[GA Chat]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        # 구글 드라이브 파일 경로 입력 받기
        google_drive_url = st.text_input("https://docs.google.com/presentation/d/1FIqHhZLqWkA0E55FGsq9getgdEG151P03HqT59L9qWs/edit#slide=id.g135ab063ed8_0_0")
        process = st.button("Process")

    if process:
        if not google_drive_url:
            st.info("Please enter a Google Drive file URL to continue.")
            st.stop()

        # 구글 드라이브에서 파일 다운로드
        files_text = download_and_get_text(google_drive_url)
        text_chunks = get_text_chunks(files_text)
        vectorestore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vectorestore, OPENAI_API_KEY)
        st.session_state.processComplete = True

    display_chat_ui()

def download_and_get_text(url):
    # 구글 드라이브 URL에서 파일 ID 추출
    file_id = url.split('/')[-2]
    output = 'downloaded_file'  # 임시 파일 이름
    gdown.download(id=file_id, output=output, quiet=False)

    # 파일 타입에 따라 적절한 로더 사용
    if output.endswith('.pdf'):
        loader = PyPDFLoader(output)
    elif output.endswith('.docx'):
        loader = Docx2txtLoader(output)
    elif output.endswith('.pptx'):
        loader = UnstructuredPowerPointLoader(output)
    else:
        st.error("Unsupported file format!")
        st.stop()

    documents = loader.load_and_split()
    return documents

# 기타 필요한 함수들 (get_text_chunks, get_vectorstore, get_conversation_chain)은 이전과 동일하게 유지

def display_chat_ui():
    # 챗봇 UI 표시 로직
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

        # Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
