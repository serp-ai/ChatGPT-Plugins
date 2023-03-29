import requests
import urllib.parse
from typing import Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import traceback
from io import StringIO
from contextlib import redirect_stdout
from duckduckgo_search import ddg, ddg_answers, ddg_images, ddg_videos, ddg_news, ddg_maps, ddg_translate, ddg_suggestions


class WebBrowser():
    def __init__(self):
        self.history = []
        self.current_url = None
        self.forward_history = []

    def open_url(self, url):
        print(f"Opening URL: {url}")
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        self.current_url = url
        if self.history and self.history[-1] != url:
            self.history.append(url)
        self.forward_history.clear()
        return soup

    def click_link(self, link_text):
        print(f"Clicking on link: {link_text}")
        soup = self.open_url(self.current_url)
        link = soup.find('a', text=link_text)
        if link:
            href = link.get('href')
            new_url = urljoin(self.current_url, href)
            soup = self.open_url(new_url)
        self.history.append(self.current_url)
        self.forward_history.clear()

    def back(self):
        print("Going back")
        if len(self.history) > 1:
            self.forward_history.append(self.history.pop())
            self.current_url = self.history[-1]
            self.open_url(self.current_url)

    def forward(self):
        print("Going forward")
        if self.forward_history:
            url = self.forward_history.pop()
            self.open_url(url)
            self.history.append(self.current_url)

    def get_readable_content(self, soup=None, url=None):
        if url:
            self.current_url = url
        if not soup:
            soup = self.open_url(self.current_url)
        content = ""
        for p in soup.find_all('p'):
            content += p.get_text()
        return {'readable_content': content}

    def get_internal_links(self, soup=None, url=None):
        if url:
            self.current_url = url
        if not soup:
            soup = self.open_url(self.current_url)
        internal_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = urljoin(self.current_url, href)
                parsed_url = urlparse(full_url)

                if parsed_url.netloc == urlparse(self.current_url).netloc:
                    relative_url = parsed_url.path
                    if relative_url not in internal_links:
                        internal_links.append(relative_url)
        return {'internal_links': internal_links}

    def get_external_links(self, soup=None, url=None):
        if url:
            self.current_url = url
        if not soup:
            soup = self.open_url(self.current_url)
        external_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = urljoin(self.current_url, href)
                parsed_url = urlparse(full_url)

                if parsed_url.netloc != urlparse(self.current_url).netloc:
                    if full_url not in external_links:
                        external_links.append(full_url)
        return {'external_links': external_links}

    def parse_page(self, soup=None, url=None):
        if url:
            self.current_url = url
        if not soup:
            soup = self.open_url(self.current_url)
        page_data = {
            "readable_content": self.get_readable_content(soup)['readable_content'],
            "internal_links": self.get_internal_links(soup)['internal_links'],
            "external_links": self.get_external_links(soup)['external_links']
        }
        return page_data


class WebSearch():
    """
    Class to handle web search
    """
    
    def __init__(self) -> None:
        """
        Initialize web search
        """
        self.cache = {}

    def search(self, keywords: Any, region: str = "us-en", safesearch: str = "Off", time: Optional[Any] = 'y', max_results: Optional[Any] = 20, page: int = 1, output: Optional[Any] = None, download: bool = False, cache: bool = False) -> list:
        """
        Search

        Parameters:
            keywords (Any): keywords
            region (str): region
            safesearch (str): safesearch
            time (Any | None): time (one of: 'd', 'w', 'm', 'y')
            max_results (Any | None): max results
            page (int): page
            output (Any | None): output
            download (bool): download
            cache (bool): If True, cache results

        Returns:
            list: results
        """
        if cache and 'search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download) in self.cache:
            return self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)]
        response = ddg(keywords=keywords, region=region, safesearch=safesearch, time=time, max_results=max_results, page=page, output=output, download=download)
        if cache:
            self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)] = response
        return response


class WolframAlpha():
    """
    Class to handle wolfram alpha api requests
    """
    def __init__(self, app_id: str) -> None:
        """
        Initialize wolfram alpha

        Parameters:
            app_id (str): app id
        """
        self.app_id = app_id
        self.cache = {}
        self.endpoints = {
            'short_answer': 'https://api.wolframalpha.com/v1/result?i=',
        }

    def get_short_answer(self, query: str, cache: bool = False) -> str:
        """
        Get short answer result

        Parameters:
            query (str): query
            cache (bool): If True, cache results

        Returns:
            str: result
        """
        if cache and 'get_short_answer' + str(query) in self.cache:
            return self.cache['get_short_answer' + str(query)]
        response = requests.get(self.endpoints['short_answer'] + urllib.parse.quote(query) + '&appid=' + self.app_id)
        response = {
            'query': query,
            'response': response.text,
        }
        if cache:
            self.cache['get_short_answer' + str(query)] = response
        return response


def python_runner(code):
    # Basic and insecure Python runner (run at your own risk!)
    # Redirect stdout to capture output
    buffer = StringIO()
    with redirect_stdout(buffer):
        try:
            # Check if the code is an expression
            is_expression = False
            try:
                compile(code, "<string>", "eval")
                is_expression = True
            except SyntaxError:
                pass

            # If the code is an expression, use eval; otherwise, use exec
            if is_expression:
                result = eval(code)
                if result is not None:
                    print(result)
            else:
                exec(code)
            return buffer.getvalue()
        except Exception as e:
            # Capture and return the traceback on error
            tb = traceback.format_exc()
            return tb
