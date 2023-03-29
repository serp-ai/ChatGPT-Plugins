# ChatGPT-Plugins
Repo for giving ChatGPT the ability to use web browsing, python code execution, and custom plugins

# How to use
- Make sure you have an openai account and have created an API key
- Open `plugins.ipynb`
- Insert your api key into the `api_key` variable in the first cell
- Run all of the setup cells
- Edit the example with your desired prompt. In the setup cell, there is a variable called `prompt` that defines the action loop and the tools to the model. So make sure that the prompt you are giving to the action loop looks like `prompt.format(prompt="Your prompt here")`

# How to edit what the model can do
There is a cell that defines a variable called prompt that looks like this:
```python
prompt = """Respond to the following prompt as best as you can. You have access to the following tools:
Web Search: Searches the web for the given search query.
Get Readable Content: Returns the readable content of the given url.
Get Internal Links: Returns the internal links of the given url.
Run Python Code: Runs the given Python code. Must be one line.
Ask ChatGPT: Ask ChatGPT a question or give it a prompt for a response.
Use the following format:
Prompt: the prompt you are responding to
Thought: you should always think about what to do
Action: the action you want to take, must be one of [Web Search, Get Readable Content, Get Internal Links, Run Python Code, Ask ChatGPT]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation loop can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Prompt: {prompt}"""
```
You can edit this prompt to add more actions, change the names of the actions, or change the descriptions.
Examples:
- Remove the `Web Search` action
```python
prompt = """Respond to the following prompt as best as you can. You have access to the following tools:
Get Readable Content: Returns the readable content of the given url.
Get Internal Links: Returns the internal links of the given url.
Run Python Code: Runs the given Python code. Must be one line.
Ask ChatGPT: Ask ChatGPT a question or give it a prompt for a response.
Use the following format:
Prompt: the prompt you are responding to
Thought: you should always think about what to do
Action: the action you want to take, must be one of [Get Readable Content, Get Internal Links, Run Python Code, Ask ChatGPT]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation loop can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Prompt: {prompt}"""
```
Don't forget to change this line to remove the `Web Search` action:
```python
Action: the action you want to take, must be one of [Get Readable Content, Get Internal Links, Run Python Code, Ask ChatGPT]
```
- Add a new action called `Calculator`
```python
prompt = """Respond to the following prompt as best as you can. You have access to the following tools:
Web Search: Searches the web for the given search query.
Get Readable Content: Returns the readable content of the given url.
Get Internal Links: Returns the internal links of the given url.
Run Python Code: Runs the given Python code. Must be one line.
Ask ChatGPT: Ask ChatGPT a question or give it a prompt for a response.
Calculator: Calculates the given basic math expression.
Use the following format:
Prompt: the prompt you are responding to
Thought: you should always think about what to do
Action: the action you want to take, must be one of [Web Search, Get Readable Content, Get Internal Links, Run Python Code, Ask ChatGPT, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation loop can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Prompt: {prompt}"""
```
Don't forget to change this line to add the `Calculator` action:
```python
Action: the action you want to take, must be one of [Web Search, Get Readable Content, Get Internal Links, Run Python Code, Ask ChatGPT, Calculator]
```
When adding an action, you also need to make the code to handle the action. There is a cell that defines a function called `parse_action`. There is an if, elif, else statement that you can add an elif to like this:
```python
if last_action.lower() == 'web search':
    ...
elif ...:
    ...
elif last_action.lower() == "calculator":
    # calculator code here
    # example
    try:
        answer = eval(last_action_input)
        content = str(answer)
        content = truncate_text(chatgpt, content.strip(), max_tokens) # truncate the text to the max tokens
        cache[last_action+last_action_input] = content # cache the content
        return content
    except:
        return "Invalid expression"
else:
    ...
```
It is important to note that you should add the truncate_text function and the cache to the end of your elif statement. The truncate_text will prevent eating up all of your tokens while the cache will prevent the model from having to recompute the same action over and over again. You do not have to change the truncate_text and cache lines, just make sure your final text is in a variable called `content`. Also try to make sure any errors are caught and returned as a string for the model to respond to.

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
