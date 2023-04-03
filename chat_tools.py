import re

from tools import python_runner, WebBrowser, WebSearch


def truncate_text(assistant, text, max_length=500, side='right'):
    if side == 'right':
        return assistant.enc.decode(assistant.enc.encode(text)[:max_length])
    else:
        return assistant.enc.decode(assistant.enc.encode(text)[-max_length:])
    

def remove_markdown_formatting(code: str) -> str:
    code = re.sub(r'(```python)|(```)|(`)', '', code)
    return code.strip()


def web_search(last_action_input, chatgpt=None, max_tokens=500):
    search = WebSearch()
    last_action_input = last_action_input.strip('\"') # remove quotes to receive better search results
    print('Searching for: ' + last_action_input + '...')
    results = search.search(keywords=last_action_input, safesearch='Off', time=None, max_results=10)
    out = '{'
    for result in results:
        out += 'title: ' + result['title'] + ',\n\tbody: ' + result['body'] + ',\n\t' + 'url: ' + result['href'] + ',\n\t'
    return truncate_text(chatgpt, out.strip(), max_tokens) + '}'


def ask_chatgpt(last_action_input, chatgpt, max_tokens):
    print('Asking ChatGPT: ' + last_action_input + '...')
    response = chatgpt.get_chat_response(last_action_input)
    return truncate_text(chatgpt, response.choices[0].message.content.strip(), max_tokens)


def get_readable_content(last_action_input, chatgpt, max_tokens):
    print(f'Getting readable content for {last_action_input}...')
    browser = WebBrowser()
    summarize_prompt = 'Summarize the following text while trying to stay under 500 words. Include all important and relevant information:\n{text}'
    max_tokens_for_prompt = 3500
    contents = str(browser.get_readable_content(url=last_action_input.strip()))
    if chatgpt.calculate_num_tokens(contents) > max_tokens:
        summary_prompt = summarize_prompt.format(text=contents)
        # Trim to max_tokens_for_prompt to add padding for the response
        summary_prompt = chatgpt.enc.decode(chatgpt.enc.encode(summary_prompt)[:max_tokens_for_prompt])
        contents = '{summarized content: ' + chatgpt.get_chat_response(summary_prompt).choices[0].message.content.strip() + '}'
    return truncate_text(chatgpt, contents[:-1], max_tokens) + '}'


def get_internal_links(last_action_input, chatgpt, max_tokens):
    browser = WebBrowser()
    contents = browser.get_internal_links(url=last_action_input.strip())['internal_links']
    return truncate_text(chatgpt, str(contents), max_tokens)[:-1] + ']'


def get_external_links(last_action_input, chatgpt, max_tokens):
    browser = WebBrowser()
    contents = browser.get_external_links(url=last_action_input.strip())['external_links']
    return truncate_text(chatgpt, str(contents), max_tokens)[:-1] + ']'


def run_python_code(last_action_input, chatgpt, max_tokens):
    # Remove markdown formatting (triple backticks) if present
    last_action_input = remove_markdown_formatting(last_action_input.strip())
    
    print(f'Running Python code: {last_action_input}...')
    return truncate_text(chatgpt, python_runner(last_action_input.strip('"')).strip(), max_tokens)