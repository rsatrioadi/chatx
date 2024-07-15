from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, CombinedMemory
from langchain.prompts import PromptTemplate
from constants import prompt_number_snippets, llm_model_to_use, llm_max_tokens
from search_indexing import search_faiss_index


class SnippetsBufferWindowMemory(ConversationBufferWindowMemory):
    """
    MemoryBuffer used to hold the document snippets. Inherits from ConversationBufferWindowMemory, and overwrites the
    load_memory_variables method
    """

    index: FAISS = None
    pages: list = []
    memory_key = 'snippets'
    snippets: list = []

    def __init__(self, *args, **kwargs):
        ConversationBufferWindowMemory.__init__(self, *args, **kwargs)
        self.index = kwargs['index']

    def load_memory_variables(self, inputs) -> dict:
        """
        Based on the user inputs, search the index and add the similar snippets to memory (but only if they aren't in the
        memory already)
        """
        similar_snippets = search_faiss_index(self.index, inputs['user_messages_history'])
        # In order to respect the buffer size and make its pruning work, need to reverse the list, and then un-reverse it later
        # This way, the most relevant snippets are kept at the start of the list
        self.snippets = [snippet for snippet in reversed(self.snippets)]
        self.pages = [page for page in reversed(self.pages)]

        for snippet in similar_snippets:
            page_number = snippet.metadata['page']
            snippet_to_add = f"The following snippet was extracted from the following document: "
            if snippet.metadata['title'] == snippet.metadata['source']:
                snippet_to_add += f"{snippet.metadata['source']}\n"
            else:
                snippet_to_add += f"[{snippet.metadata['title']}]({snippet.metadata['source']})\n"
            snippet_to_add += f"<START_SNIPPET_CHUNK_{page_number + 1}>\n"
            snippet_to_add += f"{snippet.page_content}\n"
            snippet_to_add += f"<END_SNIPPET_CHUNK_{page_number + 1}>\n"
            if snippet_to_add not in self.snippets:
                self.pages.append(page_number)
                self.snippets.append(snippet_to_add)

        # Reverse list of snippets and pages, in order to keep the most relevant at the top
        # Also prune the list to keep the buffer within the define size (k)
        self.snippets = [snippet for snippet in reversed(self.snippets)][:self.k]
        self.pages = [page for page in reversed(self.pages)][:self.k]
        to_return = ''.join(self.snippets)

        return {'snippets': to_return}


def construct_conversation(prompt: str, llm, memory) -> ConversationChain:
    """
    Construct a ConversationChain object
    """

    prompt = PromptTemplate.from_template(
        template=prompt,
    )

    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False,
        prompt=prompt
    )

    return conversation


def initialize_chat_conversation(
        index: FAISS, 
        model_to_use: str = llm_model_to_use, 
        max_tokens: int = llm_max_tokens
    ) -> ConversationChain:
    """
    Initialize a chat conversation using the provided FAISS index and model.
    """
    prompt_header = """
    <s> [INST] You are an expert research assistant specializing in the analysis of interview transcripts.
    Use the provided snippets from various documents to accurately answer the researcher's questions.
    If the snippets do not contain the answer, state that clearly without attempting to fabricate a response.
    Ensure your answers are concise, factual, and directly address the question.
    Always include the title and page number of the document where the information was found, if applicable. [/INST] </s> 
    [INST] {history} 
    Question: {input} 
    Relevant Snippets: {snippets} 
    Answer: [/INST]
    """

    llm = ChatOllama(model=model_to_use)
    conv_memory = ConversationBufferWindowMemory(k=3, input_key="input")
    snippets_memory = SnippetsBufferWindowMemory(k=prompt_number_snippets, index=index, memory_key='snippets', input_key="snippets")
    memory = CombinedMemory(memories=[conv_memory, snippets_memory])

    return construct_conversation(prompt_header, llm, memory)
