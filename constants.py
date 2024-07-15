# Number of snippets that will be added to the prompt. Too many snippets and you risk both the prompt going over the
# token limit, and the model not being able to find the correct answer
prompt_number_snippets = 20

# LLM-related constants
llm_model_to_use = "phi3:medium-128k"
# llm_model_to_use = "tinyllama:latest"
llm_max_tokens = 128*1024

# Number of past user messages that will be used to search relevant snippets
search_number_messages = 0

# Chunking constants
chunk_size = 1024
chunk_overlap = 205

# Number of snippets to be retrieved by FAISS
number_snippets_to_retrieve = 20
