import streamlit as st
from constants import search_number_messages
from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_file, load_faiss_index
import re


def remove_file(file_to_remove):
    """
    Remove file from the session_state. Triggered by the respective button.
    """
    if file_to_remove in st.session_state.uploaded_files:
        st.session_state.uploaded_files.remove(file_to_remove)


# Page title
st.set_page_config(page_title='Talk with files using LLMs - Beta')
st.title('Talk with files using LLMs - (Beta)')

# Initialize the faiss_index key in the session state
if 'faiss_index' not in st.session_state:
    st.session_state['faiss_index'] = {
        'indexed_files': [],
        'index': load_faiss_index()
    }

# Initialize conversation memory used by Langchain
if 'conversation_memory' not in st.session_state:
    st.session_state['conversation_memory'] = None

# Initialize chat history used by Streamlit (for display purposes)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Store the uploaded files added by the user in the UI
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

with st.sidebar:
    with st.form('files-form', clear_on_submit=True):
        uploaded_files = st.file_uploader('Upload relevant files:', accept_multiple_files=True)
        add_files_button = st.form_submit_button('Add')
        if add_files_button:
            if uploaded_files:
                st.session_state.uploaded_files += uploaded_files

    with st.container():
        if st.session_state.uploaded_files:
            st.header('Files added:')
            for uploaded_file in st.session_state.uploaded_files:
                st.write(uploaded_file.name)
                st.button(label='Remove', key=f"Remove {uploaded_file.name}", on_click=remove_file, kwargs={'file_to_remove': uploaded_file})
                st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if query_text := st.chat_input("Your message"):
    st.chat_message("user").markdown(query_text)
    st.session_state.messages.append({"role": "user", "content": query_text})

    session_file_names = [(file.name, file.size) for file in st.session_state.uploaded_files]
    if st.session_state['faiss_index']['index'] is None or set(st.session_state['faiss_index']['indexed_files']) != set(session_file_names):
        st.session_state['faiss_index']['indexed_files'] = session_file_names
        with st.spinner('Indexing files...'):
            faiss_index = download_and_index_file(st.session_state.uploaded_files)
            st.session_state['faiss_index']['index'] = faiss_index
    else:
        faiss_index = st.session_state['faiss_index']['index']

    if st.session_state['conversation_memory'] is None:
        conversation = initialize_chat_conversation(faiss_index)
        st.session_state['conversation_memory'] = conversation
    else:
        conversation = st.session_state['conversation_memory']

    user_messages_history = [message['content'] for message in st.session_state.messages[-search_number_messages:] if message['role'] == 'user']
    user_messages_history = '\n'.join(user_messages_history)

    with st.spinner('Querying the model...'):
        response = conversation.predict(input=query_text, user_messages_history=user_messages_history)

    with st.chat_message("assistant"):
        st.markdown(response)
        snippet_memory = conversation.memory.memories[1]
        for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
            with st.expander(f'Snippet from chunk {page_number + 1}'):
                snippet = re.sub("<START_SNIPPET_CHUNK_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_CHUNK_\d+>", '', snippet)
                st.markdown(snippet)

    st.session_state.messages.append({"role": "assistant", "content": response})
