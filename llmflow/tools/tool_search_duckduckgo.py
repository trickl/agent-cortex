"""
DuckDuckGo Search Tool for LLM Agents
=====================================

A Python tool for searching DuckDuckGo and extracting results with support
for various search options. Designed to be used as a tool in LLM agent frameworks.

Requirements:
    - selenium (install with: pip install selenium)
    - webdriver-manager (install with: pip install webdriver-manager)
    - Python 3.7+

Note: This tool uses Chrome WebDriver. Make sure Chrome is installed on your system.

Example Usage:
    >>> searcher = DuckDuckGoSearcher()
    >>> result = searcher.search("python programming")
    >>> print(result.results)

Author: AI Assistant
Version: 1.0.0
"""

import os
import re
import time
import json
import logging
import subprocess
import sys
from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import tempfile

# Check if required packages are installed
packages = {
    'selenium': 'selenium',
    'webdriver_manager': 'webdriver-manager'
}

for import_name, install_name in packages.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"{install_name} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])

# Import after installation
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator


class SearchRegion(Enum):
    """Search region options."""
    ALL = "wt-wt"  # All regions
    US = "us-en"   # United States
    UK = "uk-en"   # United Kingdom
    DE = "de-de"   # Germany
    FR = "fr-fr"   # France
    ES = "es-es"   # Spain
    IT = "it-it"   # Italy
    RU = "ru-ru"   # Russia
    CN = "cn-zh"   # China
    JP = "jp-jp"   # Japan


class SafeSearch(Enum):
    """Safe search settings."""
    OFF = "off"
    MODERATE = "moderate"
    STRICT = "strict"


class TimeRange(Enum):
    """Time range for search results."""
    ALL = ""
    DAY = "d"
    WEEK = "w"
    MONTH = "m"
    YEAR = "y"


@dataclass
class SearchOptions:
    """Configuration options for DuckDuckGo search."""
    max_results: int = 10
    region: SearchRegion = SearchRegion.ALL
    safe_search: SafeSearch = SafeSearch.MODERATE
    time_range: TimeRange = TimeRange.ALL
    headless: bool = True
    timeout: int = 30
    scroll_pause_time: float = 2.0
    max_scrolls: int = 3
    user_agent: Optional[str] = None
    proxy: Optional[str] = None
    verbose: bool = False
    extract_snippets: bool = True
    extract_urls: bool = True
    wait_for_results: int = 5
    retry_attempts: int = 3
    window_size: tuple = (1920, 1080)


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    position: int
    domain: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class SearchResponse:
    """Response from a search operation."""
    success: bool
    query: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DuckDuckGoSearcher:
    """
    A DuckDuckGo search tool for LLM agents.
    
    This class provides a high-level interface for searching DuckDuckGo
    and extracting structured results.
    
    Attributes:
        options (SearchOptions): Configuration options for searching
        logger (logging.Logger): Logger instance for this class
        driver: Selenium WebDriver instance
    """
    
    def __init__(
        self,
        options: Optional[SearchOptions] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the DuckDuckGo searcher.
        
        Args:
            options: Search configuration options
            logger: Logger instance (creates default if None)
        """
        self.options = options or SearchOptions()
        self.logger = logger or self._setup_logger()
        self.driver = None
        self._temp_dir = tempfile.mkdtemp()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO if not self.options.verbose else logging.DEBUG)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with options."""
        chrome_options = Options()
        
        if self.options.headless:
            chrome_options.add_argument("--headless")
        
        # Set window size
        chrome_options.add_argument(f"--window-size={self.options.window_size[0]},{self.options.window_size[1]}")
        
        # Set user agent if provided or use a realistic one
        if self.options.user_agent:
            chrome_options.add_argument(f"user-agent={self.options.user_agent}")
        else:
            chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        # Set proxy if provided
        if self.options.proxy:
            chrome_options.add_argument(f"--proxy-server={self.options.proxy}")
        
        # Additional options for stability
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-gpu")
        
        # Anti-bot detection measures
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Initialize driver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # Execute CDP commands to prevent detection
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            """
        })
        
        driver.set_page_load_timeout(self.options.timeout)
        
        return driver
    
    def _extract_search_result(self, element, position: int) -> Optional[SearchResult]:
        """
        Extract search result from a web element.
        
        Args:
            element: Selenium web element
            position: Result position
            
        Returns:
            SearchResult or None if extraction fails
        """
        try:
            # Get full text
            full_text = element.text
            if not full_text:
                return None
            
            lines = full_text.split('\n')
            
            # Extract title (usually first line)
            title = lines[0] if lines else "No title"
            
            # Extract URL
            url = None
            
            # Try to find <a> elements with different approaches
            # First try direct links
            a_elements = element.find_elements(By.TAG_NAME, "a")
            for a in a_elements:
                href = a.get_attribute("href")
                if href and not href.startswith("javascript") and "duckduckgo.com" not in href:
                    url = href
                    break
                    
            # Try specific selectors for links if not found
            if not url:
                link_selectors = [
                    "a.result__a", 
                    "a.result-link", 
                    "a.result__url", 
                    "a[data-testid='result-title-a']",
                    "a.eVNpHGjtxRBq_gLOfGDr",  # New DuckDuckGo class
                    "a[data-testid='result-title-link']"
                ]
                
                for selector in link_selectors:
                    try:
                        link_element = element.find_element(By.CSS_SELECTOR, selector)
                        href = link_element.get_attribute("href")
                        if href and not href.startswith("javascript") and "duckduckgo.com" not in href:
                            url = href
                            break
                    except NoSuchElementException:
                        continue
                    
            # If no URL found, try other methods
            if not url:
                # Check data attributes
                for attr in ["data-href", "data-url"]:
                    potential_url = element.get_attribute(attr)
                    if potential_url:
                        url = potential_url
                        break
            
            # Extract domain
            domain = None
            if url:
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                if domain_match:
                    domain = domain_match.group(1)
            
            # Extract snippet
            snippet_parts = []
            url_pattern = re.compile(r'^(https?://|www\.)')
            
            for line in lines[1:]:
                if line and not url_pattern.match(line) and line != title:
                    # Skip short timestamps or domain names
                    if not re.match(r'^\d+[hmd]$', line.strip()) and len(line.strip()) > 2:
                        if domain and line.strip() == domain:
                            continue
                        snippet_parts.append(line)
            
            snippet = " ".join(snippet_parts) if snippet_parts else "No snippet available"
            
            return SearchResult(
                title=title,
                url=url or "URL not found",
                snippet=snippet,
                position=position,
                domain=domain,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.debug(f"Error extracting result: {str(e)}")
            return None
    
    def search(
        self,
        query: str,
        options: Optional[SearchOptions] = None
    ) -> SearchResponse:
        """
        Search DuckDuckGo for the given query.
        
        Args:
            query: Search query
            options: Optional search options (overrides instance options)
            
        Returns:
            SearchResponse object with results
        """
        # Use provided options or instance options
        if options:
            original_options = self.options
            self.options = options
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Searching for: {query}")
            
            # Set up driver if not already done
            if not self.driver:
                self.driver = self._setup_driver()
            
            # Navigate to DuckDuckGo
            self.driver.get("https://duckduckgo.com/")
            
            # Find search box
            search_box = None
            for selector in ["searchbox_input", "q"]:
                try:
                    if selector == "searchbox_input":
                        search_box = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.ID, selector))
                        )
                    else:
                        search_box = WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.NAME, selector))
                        )
                    break
                except TimeoutException:
                    continue
            
            if not search_box:
                raise TimeoutException("Could not find search box")
            
            # Enter search query
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            
            # Wait for results with a more dynamic approach
            try:
                WebDriverWait(self.driver, self.options.wait_for_results).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".react-results--main, .serp__results, article, .result"))
                )
            except TimeoutException:
                self.logger.warning("Timed out waiting for search results to appear")
                # Continue anyway as we'll try different selectors
                
            # Scroll to load more results
            for i in range(self.options.max_scrolls):
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(self.options.scroll_pause_time)
            
            # Scroll back to top
            self.driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)
            
            # Extract results
            results = []
            
            # Try different selectors (updated for latest DuckDuckGo structure)
            result_selectors = [
                "article",
                ".result__body",
                ".nrn-react-div article",
                ".result",
                "div[data-testid='result']",
                ".react-results--main .react-results--result",
                ".react-results--main article",
                ".react-results--main .result",
                ".react-results--main .web-result",
                ".web-result"
            ]
            
            search_elements = []
            for selector in result_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        search_elements = elements
                        self.logger.debug(f"Found {len(elements)} results with selector: {selector}")
                        break
                except:
                    continue
            
            # Process results
            for i, element in enumerate(search_elements[:self.options.max_results], 1):
                result = self._extract_search_result(element, i)
                if result:
                    results.append(result)
                    
            # Debug output if no results found
            if not results:
                self.logger.warning(f"No results extracted. Found {len(search_elements)} potential result elements.")
                self.logger.debug(f"Page source preview: {self.driver.page_source[:500]}...")
                
            search_time = time.time() - start_time
            
            return SearchResponse(
                success=True,
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                metadata={
                    'selector_used': selector if search_elements else None,
                    'elements_found': len(search_elements)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return SearchResponse(
                success=False,
                query=query,
                error_message=f"Search error: {str(e)}",
                search_time=time.time() - start_time
            )
        
        finally:
            # Restore original options if they were overridden
            if options:
                self.options = original_options
    
    def search_with_retry(
        self,
        query: str,
        max_retries: Optional[int] = None,
        options: Optional[SearchOptions] = None
    ) -> SearchResponse:
        """
        Search with automatic retry on failure.
        
        Args:
            query: Search query
            max_retries: Maximum retry attempts
            options: Optional search options
            
        Returns:
            SearchResponse object
        """
        max_retries = max_retries or self.options.retry_attempts
        last_error = None
        
        for attempt in range(max_retries + 1):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt}/{max_retries}")
                # Reset driver on retry
                self.close()
            
            response = self.search(query, options)
            
            if response.success:
                return response
            
            last_error = response.error_message
        
        return SearchResponse(
            success=False,
            query=query,
            error_message=f"Failed after {max_retries} retries. Last error: {last_error}"
        )
    
    def close(self):
        """Close the browser and clean up resources."""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class DuckDuckGoSearchTool:
    """
    LLM Agent Tool wrapper for DuckDuckGo Search.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, default_options: Optional[SearchOptions] = None):
        """
        Initialize the tool.
        
        Args:
            default_options: Default search options
        """
        self.searcher = DuckDuckGoSearcher(default_options)
        self.name = "duckduckgo_search"
        self.description = (
            "Search DuckDuckGo for web results. "
            "Input should be a search query. "
            "Returns structured search results with titles, URLs, and snippets."
        )
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run the search tool.
        
        Args:
            query: Search query
            **kwargs: Additional options (max_results, headless, etc.)
            
        Returns:
            Dictionary with search results
        """
        # Parse kwargs into options
        options = SearchOptions()
        
        if kwargs.get('max_results'):
            options.max_results = kwargs['max_results']
        
        if kwargs.get('headless') is not None:
            options.headless = kwargs['headless']
        
        if kwargs.get('region'):
            region_map = {
                'us': SearchRegion.US,
                'uk': SearchRegion.UK,
                'de': SearchRegion.DE,
                'fr': SearchRegion.FR,
                'all': SearchRegion.ALL
            }
            options.region = region_map.get(kwargs['region'].lower(), SearchRegion.ALL)
        
        # Search
        response = self.searcher.search_with_retry(query, options=options)
        
        # Convert to dictionary for agent compatibility
        return {
            'success': response.success,
            'query': response.query,
            'results': [
                {
                    'title': r.title,
                    'url': r.url,
                    'snippet': r.snippet,
                    'domain': r.domain,
                    'position': r.position
                }
                for r in response.results
            ],
            'total_results': response.total_results,
            'search_time': response.search_time,
            'error': response.error_message
        }
    
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Async version of run (currently just calls sync version)."""
        return self.run(query, **kwargs)
    
    def cleanup(self):
        """Clean up resources."""
        self.searcher.close()


@register_tool(
    tags=["search", "web", "duckduckgo", "research"]
)
def search_duckduckgo(
    query: str,
    max_results: int = 10,
    headless: bool = True
) -> Dict[str, Any]:
    """Searches DuckDuckGo and returns structured results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        headless: Run browser in headless mode
        
    Returns:
        Dictionary with search results
    """
    options = SearchOptions(
        max_results=max_results,
        headless=headless
    )
    
    with DuckDuckGoSearcher(options) as searcher:
        response = searcher.search(query)
        
        return {
            "success": response.success,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "domain": r.domain
                }
                for r in response.results
            ],
            "total": response.total_results,
            "error": response.error_message
        }


@register_tool(
    tags=["search", "web", "duckduckgo", "quick"]
)
def quick_search(query: str) -> List[str]:
    """Performs a quick DuckDuckGo search and returns just the URLs.
    
    Args:
        query: The search query
        
    Returns:
        List of result URLs
    """
    options = SearchOptions(max_results=5, headless=True)
    
    with DuckDuckGoSearcher(options) as searcher:
        response = searcher.search(query)
        
        if response.success:
            return [r.url for r in response.results if r.url != "URL not found"]
        else:
            return []


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Basic search
    searcher = DuckDuckGoSearcher()
    result = searcher.search("Python programming tutorials")
    
    if result.success:
        print(f"Found {result.total_results} results for '{result.query}'")
        print(f"Search time: {result.search_time:.2f} seconds\n")
        
        for r in result.results:
            print(f"Result {r.position}:")
            print(f"Title: {r.title}")
            print(f"URL: {r.url}")
            print(f"Domain: {r.domain}")
            print(f"Snippet: {r.snippet}")
            print("-" * 80)
    else:
        print(f"Search failed: {result.error_message}")
    
    # Clean up
    searcher.close()
    
    # Example 2: Using as LLM tool
    print("\n\n=== LLM Tool Example ===")
    tool = DuckDuckGoSearchTool()
    tool_result = tool.run(
        "latest AI news 2024",
        max_results=3,
        headless=True
    )
    
    print(f"Tool result: {json.dumps(tool_result, indent=2)}")
    tool.cleanup()
    
    # Example 3: Using context manager
    print("\n\n=== Context Manager Example ===")
    with DuckDuckGoSearcher(SearchOptions(max_results=3)) as searcher:
        response = searcher.search("machine learning courses")
        if response.success:
            for r in response.results:
                print(f"- {r.title}: {r.url}")