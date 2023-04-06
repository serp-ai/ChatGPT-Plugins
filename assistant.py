import tiktoken
import openai
from datetime import datetime
from typing import Any
from time import sleep

from memory_manager import MemoryManager


class OpenAIAssistant():
    """
    ChatGPT wrapper for OpenAI API
    """
    def __init__(
            self,
            api_key: str, 
            chat_model: str = 'gpt-3.5-turbo', 
            embedding_model: Any = 'text-embedding-ada-002', 
            enc: str = 'gpt2', 
            short_term_memory_summary_prompt: str = None, 
            long_term_memory_summary_prompt: str = None, 
            system_prompt: str = "You are a helpful assistant. Your name is SERPy.", 
            short_term_memory_max_tokens: int = 750, 
            long_term_memory_max_tokens: int = 500,
            knowledge_retrieval_max_tokens: int = 1000,
            short_term_memory_summary_max_tokens: int = 300, 
            long_term_memory_summary_max_tokens: int = 300,
            knowledge_retrieval_summary_max_tokens: int = 600,
            summarize_short_term_memory: bool = False,
            summarize_long_term_memory: bool = False,
            summarize_knowledge_retrieval: bool = False,
            use_long_term_memory: bool = False,
            long_term_memory_collection_name: str = 'long_term_memory', 
            use_short_term_memory: bool = False, 
            use_knowledge_retrieval: bool = False,
            knowledge_retrieval_collection_name: str = 'knowledge_retrieval',
            price_per_token: float = 0.000002, 
            max_seq_len: int = 4096, 
            memory_manager: MemoryManager = None,
            debug: bool = False
        ) -> None:
        """
        Initialize the OpenAIAssistant

        Parameters:
            api_key (str): The OpenAI API key
            chat_model (str): The model to use for chat
            embedding_model (Any): The model to use for embeddings
            enc (str): The encoding to use for the model
            short_term_memory_summary_prompt (str): The prompt to use for short term memory summarization
            long_term_memory_summary_prompt (str): The prompt to use for long term memory summarization
            system_prompt (str): The system prompt to use for the model
            short_term_memory_max_tokens (int): The maximum number of tokens to store in short term memory
            long_term_memory_max_tokens (int): The maximum number of tokens to store in long term memory
            knowledge_retrieval_max_tokens (int): The maximum number of tokens to store in knowledge retrieval
            short_term_memory_summary_max_tokens (int): The maximum number of tokens to store in short term memory summary
            long_term_memory_summary_max_tokens (int): The maximum number of tokens to store in long term memory summary
            knowledge_retrieval_summary_max_tokens (int): The maximum number of tokens to store in knowledge retrieval summary
            summarize_short_term_memory (bool): Whether to use short term memory summarization
            summarize_long_term_memory (bool): Whether to use long term memory summarization
            summarize_knowledge_retrieval (bool): Whether to use knowledge retrieval summarization
            use_long_term_memory (bool): Whether to use long term memory
            long_term_memory_collection_name (str): The name of the long term memory collection
            use_short_term_memory (bool): Whether to use short term memory
            use_knowledge_retrieval (bool): Whether to use knowledge retrieval
            knowledge_retrieval_collection_name (str): The name of the knowledge retrieval collection
            price_per_token (float): The price per token in USD
            max_seq_len (int): The maximum sequence length
            memory_manager (MemoryManager): The memory manager to use for long term memory and knowledge retrieval
            debug (bool): Whether to enable debug mode
        """
        openai.api_key = api_key
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.enc = tiktoken.get_encoding(enc)
        self.memory_manager = memory_manager
        self.price_per_token = price_per_token
        self.short_term_memory = []
        self.short_term_memory_summary = ''
        self.long_term_memory_summary = ''
        self.knowledge_retrieval_summary = ''
        self.debug = debug

        self.summarize_short_term_memory = summarize_short_term_memory
        self.summarize_long_term_memory = summarize_long_term_memory
        self.summarize_knowledge_retrieval = summarize_knowledge_retrieval
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory_collection_name = 'long_term_memory' if long_term_memory_collection_name is None else long_term_memory_collection_name
        self.use_knowledge_retrieval = use_knowledge_retrieval
        self.knowledge_retrieval_collection_name = 'knowledge_retrieval' if knowledge_retrieval_collection_name is None else knowledge_retrieval_collection_name
        if self.memory_manager is None:
            self.use_long_term_memory = False
            self.use_knowledge_retrieval = False
        if self.use_long_term_memory and self.memory_manager is not None:
            self.memory_manager.create_collection(self.long_term_memory_collection_name)
        if self.use_knowledge_retrieval and self.memory_manager is not None:
            self.memory_manager.create_collection(self.knowledge_retrieval_collection_name)
        self.use_short_term_memory = use_short_term_memory

        self.short_term_memory_summary_max_tokens = short_term_memory_summary_max_tokens
        self.long_term_memory_summary_max_tokens = long_term_memory_summary_max_tokens
        self.knowledge_retrieval_summary_max_tokens = knowledge_retrieval_summary_max_tokens
        self.short_term_memory_max_tokens = short_term_memory_max_tokens
        self.long_term_memory_max_tokens = long_term_memory_max_tokens
        self.knowledge_retrieval_max_tokens = knowledge_retrieval_max_tokens

        self.system_prompt = system_prompt
        if short_term_memory_summary_prompt is None:
            self.short_term_memory_summary_prompt = "Summarize the following conversation:\n\nPrevious Summary: {previous_summary}\n\nConversation: {conversation}"
        else:
            self.short_term_memory_summary_prompt = short_term_memory_summary_prompt
        if long_term_memory_summary_prompt is None:
            self.long_term_memory_summary_prompt = "Summarize the following (out of order) conversation messages:\n\nPrevious Summary: {previous_summary}\n\nMessages: {conversation}"

        self.max_seq_len = max_seq_len

    def _construct_messages(self, prompt: str, inject_messages: list = []) -> list:
        """
        Construct the messages for the chat completion

        Parameters:
            prompt (str): The prompt to construct the messages for
            inject_messages (list): The messages to inject into the chat completion

        Returns:
            list: The messages to use for the chat completion
        """
        messages = []
        if self.system_prompt is not None and self.system_prompt != "":
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        if self.use_long_term_memory:
            long_term_memory = self.query_long_term_memory(prompt, summarize=self.summarize_long_term_memory)
            if long_term_memory is not None and long_term_memory != '':
                messages.append({
                    "role": "system",
                    "content": long_term_memory
                })

        if self.summarize_short_term_memory:
            if self.short_term_memory_summary != '' and self.short_term_memory_summary is not None:
                messages.append({
                    "role": "system",
                    "content": self.short_term_memory_summary
                })

        if self.use_short_term_memory:
            for i, message in enumerate(self.short_term_memory):
                messages.append(message)

        if inject_messages is not None and inject_messages != []:
            for i in range(len(messages)):
                for y, message in enumerate(inject_messages):
                    if i == list(message.keys())[0]:
                        messages.insert(i, list(message.values())[0])
                        inject_messages.pop(y)
            for message in inject_messages:
                messages.append(list(message.values())[0])

        if prompt is None or prompt == "":
            return messages
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages

    def change_system_prompt(self, system_prompt: str) -> None:
        """
        Change the system prompt

        Parameters:
            system_prompt (str): The new system prompt to use
        """
        self.system_prompt = system_prompt

    def calculate_num_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens in a given text

        Parameters:
            text (str): The text to calculate the number of tokens for

        Returns:
            int: The number of tokens in the text
        """
        return len(self.enc.encode(text))

    def calculate_short_term_memory_tokens(self) -> int:
        """
        Calculate the number of tokens in short term memory

        Returns:
            int: The number of tokens in short term memory
        """
        return sum([self.calculate_num_tokens(message['content']) for message in self.short_term_memory])
    
    def query_long_term_memory(self, query: str, summarize=False) -> str:
        """
        Query long term memory

        Parameters:
            query (str): The query to use for long term memory
            summarize (bool): Whether to summarize the long term memory

        Returns:
            str: The long term memory
        """
        embedding = self.get_embedding(query).data[0].embedding
        points = self.memory_manager.search_points(vector=embedding, collection_name=self.long_term_memory_collection_name, k=20)
        if len(points) == 0:
            return ''
        long_term_memory = ''
        if summarize:
            long_term_memory += 'Summary of previous related conversations from long term memory:' + self.generate_long_term_memory_summary(points) + '\n\n'
        if self.long_term_memory_max_tokens > 0:
            long_term_memory += 'Previous related conversations from long term memory:\n\n'
            for point in points:
                point = point.payload
                if self.calculate_num_tokens(long_term_memory + f"{point['user_message']['role'].title()}: {point['user_message']['content']}\n\n{point['assistant_message']['role'].title()}: {point['assistant_message']['content']}\n----------\n") > self.long_term_memory_max_tokens:
                    continue
                long_term_memory += f"{point['user_message']['role'].title()}: {point['user_message']['content']}\n\n{point['assistant_message']['role'].title()}: {point['assistant_message']['content']}\n----------\n"
        if long_term_memory == 'Previous related conversations from long term memory:\n\n':
            return ''
        elif long_term_memory.endswith('\n\nPrevious related conversations from long term memory:\n\n'):
            long_term_memory = long_term_memory.replace('\n\nPrevious related conversations from long term memory:\n\n', '')
        return long_term_memory.strip()

    def add_message_to_short_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to short term memory

        Parameters:
            user_message (dict): The user message to add to short term memory
            assistant_message (dict): The assistant message to add to short term memory
        """
        self.short_term_memory.append(user_message)
        self.short_term_memory.append(assistant_message)
        while self.calculate_short_term_memory_tokens() > self.short_term_memory_max_tokens:
            if self.summarize_short_term_memory:
                self.generate_short_term_memory_summary()
            self.short_term_memory.pop(0) # Remove the oldest message (User message)
            self.short_term_memory.pop(0) # Remove the oldest message (OpenAIAssistant message)

    def add_message_to_long_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to long term memory

        Parameters:
            user_message (dict): The user message to add to long term memory
            assistant_message (dict): The assistant message to add to long term memory
        """
        points = [
            {
                "vector": self.get_embedding(f'User: {user_message["content"]}\n\nAssistant: {assistant_message["content"]}').data[0].embedding,
                "payload": {
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "timestamp": datetime.now().timestamp()
                }
            }
        ]
        self.memory_manager.insert_points(collection_name=self.long_term_memory_collection_name, points=points)

    def generate_short_term_memory_summary(self) -> None:
        """
        Generate a summary of short term memory
        """
        prompt = self.short_term_memory_summary_prompt.format(
            previous_summary=self.short_term_memory_summary,
            conversation=f'User: {self.short_term_memory[0]["content"]}\n\nAssistant: {self.short_term_memory[1]["content"]}'
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.short_term_memory_summary_max_tokens:
            prompt = self.enc.decode(self.enc.encode(prompt)[:self.max_seq_len - self.short_term_memory_summary_max_tokens])
        summary_agent = OpenAIAssistant(self.api_key, system_prompt=None)
        self.short_term_memory_summary = summary_agent.get_chat_response(prompt, max_tokens=self.short_term_memory_summary_max_tokens).choices[0].message.content

    def generate_long_term_memory_summary(self, points: list) -> str:
        """
        Summarize long term memory

        Parameters:
            points (list): The points to summarize

        Returns:
            str: The summary of long term memory
        """
        prompt = self.long_term_memory_summary_prompt.format(
            previous_summary=self.long_term_memory_summary,
            conversation='\n\n'.join([f'User: {point.payload["user_message"]["content"]}\n\nAssistant: {point.payload["assistant_message"]["content"]}' for point in points])
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.long_term_memory_summary_max_tokens:
            prompt = self.enc.decode(self.enc.encode(prompt)[:self.max_seq_len - self.long_term_memory_summary_max_tokens])
        summary_agent = OpenAIAssistant(self.api_key, system_prompt=None)
        self.long_term_memory_summary = summary_agent.get_chat_response(prompt, max_tokens=self.long_term_memory_summary_max_tokens).choices[0].message.content
        return self.long_term_memory_summary

    def calculate_price(self, prompt: str = None, num_tokens: int = None) -> float:
        """
        Calculate the price of a prompt (or number of tokens) in USD

        Parameters:
            prompt (str): The prompt to calculate the price of
            num_tokens (int): The number of tokens to calculate the price of

        Returns:
            float: The price of the generation in USD
        """
        assert prompt or num_tokens, "You must provide either a prompt or number of tokens"
        if prompt:
            num_tokens = self.calculate_num_tokens(prompt)
        return num_tokens * self.price_per_token

    def get_embedding(self, input: str, user: str = '', instructor_instruction: str = None) -> str:
        """
        Get the embedding for given text

        Parameters:
            input (str): The text to get the embedding for
            user (str): The user to get the embedding for
            instructor_instruction (str): The instructor instruction to get the embedding with

        Returns:
            str: The embedding for the prompt
        """
        if self.embedding_model is None:
            return None
        elif self.embedding_model == 'text-embedding-ada-002':
            return openai.Embedding.create(
                model=self.embedding_model,
                input=input,
                user=user
            )
        else:
            if instructor_instruction is not None:
                return self.embedding_model.encode([[instructor_instruction, input]])
            return self.embedding_model.encode([input])

    def get_chat_response(self, prompt: str, max_tokens: int = None, temperature: float = 1.0, top_p: float = 1.0, n: int = 1, stream: bool = False, frequency_penalty: float = 0, presence_penalty: float = 0, stop: list = None, logit_bias: dict = {}, user: str = '', max_retries: int = 3, inject_messages: list = []) -> str:
        """
        Get a chat response from the model

        Parameters:
            prompt (str): The prompt to generate a response for
            max_tokens (int): The maximum number of tokens to generate
            temperature (float): The temperature of the model
            top_p (float): The top_p of the model
            n (int): The number of responses to generate
            stream (bool): Whether to stream the response
            frequency_penalty (float): The frequency penalty of the model
            presence_penalty (float): The presence penalty of the model
            stop (list): The stop sequence of the model
            logit_bias (dict): The logit bias of the model
            user (str): The user to generate the response for
            max_retries (int): The maximum number of retries to generate a response
            inject_messages (list): The messages to inject into the prompt (key: index to insert at in short term memory (0 to prepend before all messages), value: message to inject)

        Returns:
            str: The chat response
        """
        messages = self._construct_messages(prompt, inject_messages=inject_messages)
        if self.debug:
            print(f'Messages: {messages}')

        iteration = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.chat_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stream=stream,
                    stop=stop,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    user=user
                )

                if self.use_short_term_memory:
                    self.add_message_to_short_term_memory(user_message={
                        "role": "user",
                        "content": prompt
                    }, assistant_message=response.choices[0].message.to_dict())

                if self.use_long_term_memory:
                    self.add_message_to_long_term_memory(user_message={
                        "role": "user",
                        "content": prompt
                    }, assistant_message=response.choices[0].message.to_dict())

                return response
            except Exception as e:
                iteration += 1
                if iteration >= max_retries:
                    raise e
                print('Error communicating with chatGPT:', e)
                sleep(1)
