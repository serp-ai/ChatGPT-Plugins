# ChatGPT-Plugins
Repo for giving ChatGPT the ability to use web browsing, python code execution, and custom plugins

# How to use
- Make sure you have an openai account and have created an API key
- Open `plugins.ipynb`
- Insert your api key into the `api_key` variable in the first cell
- Run all of the setup cells
- Edit the example with your desired prompt.

# How to edit what the model can do
- Create a function that takes the arguments (last_action_input, chatgpt, max_tokens) and outputs the result of the truncate_text function (Don't forget to validate the input string matches what your function logic expects) (example in notebook)
- Add the function `name`, `definition`, and the `function` as a dictionary to the `function_definitions` list (example in notebook)

# How to use assistant class
- The assistant class is a wrapper around chat style openai models. It has support for short term memory, long term memory, knowledge retrieval, memory summarization, and more. In order to use the action loop, make sure you do not use short or long term memory. If you want to use long term memory, you need to set up a docker container for (Qdrant)[https://qdrant.tech/] (free).
- You can set it up like:
```python
api_key = ''
system_prompt = None
debug = False
use_long_term_memory = False
use_short_term_memory = False
use_knowledge_retrieval = False
summarize_short_term_memory = False
summarize_long_term_memory = False
summarize_knowledge_retrieval = False
short_term_memory_max_tokens = 750
long_term_memory_max_tokens = 500
knowledge_retrieval_max_tokens = 1000
short_term_memory_summary_max_tokens = 300
long_term_memory_summary_max_tokens = 300
knowledge_retrieval_summary_max_tokens = 600
long_term_memory_collection_name = 'long_term_memory'

assistant = OpenAIAssistant(api_key, system_prompt=system_prompt, long_term_memory_collection_name=long_term_memory_collection_name, use_long_term_memory=use_long_term_memory, use_short_term_memory=use_short_term_memory, memory_manager=None, debug=debug, summarize_short_term_memory=summarize_short_term_memory, summarize_long_term_memory=summarize_long_term_memory, short_term_memory_max_tokens=short_term_memory_max_tokens, long_term_memory_max_tokens=long_term_memory_max_tokens, short_term_memory_summary_max_tokens=short_term_memory_summary_max_tokens, long_term_memory_summary_max_tokens=long_term_memory_summary_max_tokens, use_knowledge_retrieval=use_knowledge_retrieval, summarize_knowledge_retrieval=summarize_knowledge_retrieval, knowledge_retrieval_max_tokens=knowledge_retrieval_max_tokens, knowledge_retrieval_summary_max_tokens=knowledge_retrieval_summary_max_tokens)
```
- You can then use the assistant like:
```python
assistant.get_chat_response(prompt)
```
Depending on your settings, the class will handle the short term memory, long term memory, memory summarization, and messages contruction.
