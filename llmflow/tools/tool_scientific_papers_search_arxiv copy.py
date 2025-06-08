"""
arXiv Search Tool for LLM Agents
=================================

A Python tool for searching and downloading papers from arXiv.org.
Designed to be used as a tool in LLM agent frameworks.

Requirements:
    - arxiv (install with: pip install arxiv)
    - requests (install with: pip install requests)
    - Python 3.7+

Example Usage:
    >>> searcher = ArxivSearcher()
    >>> results = searcher.search("quantum computing", max_results=5)
    >>> paper = searcher.get_paper("2301.08727")
    >>> searcher.download_pdf("2301.08727", "paper.pdf")

Author: AI Assistant
Version: 1.0.0
"""

import os
import re
import json
import logging
import requests
import subprocess
import sys
import time
from typing import Dict, Optional, Union, List, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlencode

try:
    from .tool_decorator import register_tool
except ImportError:
    # If tool_decorator doesn't exist, create a dummy decorator
    def register_tool(tags=None):
        def decorator(func):
            return func
        return decorator

# Check if arxiv is installed
try:
    import arxiv
except ImportError:
    print("arxiv is not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "arxiv"])
    import arxiv


class SortCriterion(Enum):
    """Sort criteria for arXiv search results."""
    RELEVANCE = "relevance"
    LAST_UPDATED = "lastUpdatedDate"
    SUBMITTED = "submittedDate"


class SortOrder(Enum):
    """Sort order for search results."""
    ASCENDING = "ascending"
    DESCENDING = "descending"


class ArxivCategory(Enum):
    """Major arXiv categories."""
    # Physics
    PHYSICS = "physics"
    ASTROPHYSICS = "astro-ph"
    CONDENSED_MATTER = "cond-mat"
    GR_QC = "gr-qc"  # General Relativity and Quantum Cosmology
    HEP = "hep"  # High Energy Physics
    QUANTUM_PHYSICS = "quant-ph"
    
    # Mathematics
    MATHEMATICS = "math"
    
    # Computer Science
    CS = "cs"
    CS_AI = "cs.AI"  # Artificial Intelligence
    CS_LG = "cs.LG"  # Machine Learning
    CS_CV = "cs.CV"  # Computer Vision
    CS_CL = "cs.CL"  # Computation and Language
    CS_CR = "cs.CR"  # Cryptography and Security
    CS_DS = "cs.DS"  # Data Structures and Algorithms
    CS_NE = "cs.NE"  # Neural and Evolutionary Computing
    
    # Quantitative Biology
    Q_BIO = "q-bio"
    
    # Quantitative Finance
    Q_FIN = "q-fin"
    
    # Statistics
    STAT = "stat"
    STAT_ML = "stat.ML"  # Machine Learning
    
    # Electrical Engineering
    EESS = "eess"  # Electrical Engineering and Systems Science
    
    # Economics
    ECON = "econ"


@dataclass
class SearchOptions:
    """Configuration options for arXiv search."""
    max_results: int = 10
    sort_by: SortCriterion = SortCriterion.RELEVANCE
    sort_order: SortOrder = SortOrder.DESCENDING
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    categories: List[str] = field(default_factory=list)
    include_abstract: bool = True
    include_comments: bool = True
    include_journal_ref: bool = True
    include_authors: bool = True
    include_pdf_url: bool = True
    download_pdfs: bool = False
    pdf_dir: str = "./arxiv_papers"


@dataclass
class Author:
    """Author information."""
    name: str
    affiliations: List[str] = field(default_factory=list)


@dataclass
class Paper:
    """Represents an arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[Author]
    abstract: str
    published: datetime
    updated: datetime
    categories: List[str]
    primary_category: str
    pdf_url: str
    entry_url: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary."""
        data = asdict(self)
        # Convert datetime objects to strings
        data['published'] = self.published.isoformat()
        data['updated'] = self.updated.isoformat()
        return data


@dataclass
class SearchResult:
    """Result of an arXiv search."""
    success: bool
    query: str
    total_results: int
    papers: List[Paper] = field(default_factory=list)
    error_message: Optional[str] = None
    search_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArxivSearcher:
    """
    An arXiv search tool for LLM agents.
    
    This class provides a high-level interface for searching arXiv papers,
    retrieving paper details, and downloading PDFs.
    
    Attributes:
        options (SearchOptions): Configuration options for searching
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(
        self,
        options: Optional[SearchOptions] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the arXiv searcher.
        
        Args:
            options: Search configuration options
            logger: Logger instance (creates default if None)
        """
        self.options = options or SearchOptions()
        self.logger = logger or self._setup_logger()
        self._ensure_pdf_directory()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up default logger configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _ensure_pdf_directory(self) -> None:
        """Ensure the PDF directory exists."""
        if self.options.download_pdfs:
            Path(self.options.pdf_dir).mkdir(parents=True, exist_ok=True)
    
    def _build_query(
        self,
        query: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        categories: Optional[List[str]] = None,
        arxiv_id: Optional[str] = None
    ) -> str:
        """
        Build an arXiv query string.
        
        Args:
            query: General search query
            title: Search in title
            author: Search by author
            abstract: Search in abstract
            categories: Filter by categories
            arxiv_id: Specific arXiv ID
            
        Returns:
            Formatted query string
        """
        if arxiv_id:
            return f"id:{arxiv_id}"
        
        query_parts = []
        
        if query and not (title or author or abstract):
            # General query - search all fields
            query_parts.append(f"all:{query}")
        else:
            # Specific field queries
            if title:
                query_parts.append(f"ti:{title}")
            if author:
                query_parts.append(f"au:{author}")
            if abstract:
                query_parts.append(f"abs:{abstract}")
            if query and (title or author or abstract):
                # Add general query as well
                query_parts.append(f"all:{query}")
        
        # Add category filters
        if categories or self.options.categories:
            cats = categories or self.options.categories
            cat_query = " OR ".join([f"cat:{cat}" for cat in cats])
            if cat_query:
                query_parts.append(f"({cat_query})")
        
        # Combine with AND
        full_query = " AND ".join(query_parts) if query_parts else "all:*"
        
        return full_query
    
    def _parse_paper(self, result: arxiv.Result) -> Paper:
        """Parse arxiv.Result into Paper object."""
        # Extract authors
        authors = []
        for author in result.authors:
            authors.append(Author(
                name=str(author),
                affiliations=[]  # arxiv API doesn't provide affiliations
            ))
        
        # Create Paper object
        return Paper(
            arxiv_id=result.entry_id.split('/')[-1],
            title=result.title,
            authors=authors,
            abstract=result.summary,
            published=result.published,
            updated=result.updated,
            categories=result.categories,
            primary_category=result.primary_category,
            pdf_url=result.pdf_url,
            entry_url=result.entry_id,
            comment=result.comment,
            journal_ref=result.journal_ref,
            doi=result.doi
        )
    
    def search(
        self,
        query: str,
        title: Optional[str] = None,
        author: Optional[str] = None,
        abstract: Optional[str] = None,
        categories: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        options: Optional[SearchOptions] = None
    ) -> SearchResult:
        """
        Search for papers on arXiv.
        
        Args:
            query: General search query
            title: Search in title
            author: Search by author
            abstract: Search in abstract
            categories: Filter by categories
            max_results: Maximum number of results
            options: Optional search options (overrides instance options)
            
        Returns:
            SearchResult object with found papers
        """
        # Use provided options or instance options
        opts = options or self.options
        max_results = max_results or opts.max_results
        
        # Record start time
        start_time = time.time()
        
        try:
            # Build query
            arxiv_query = self._build_query(
                query, title, author, abstract, categories
            )
            
            self.logger.info(f"Searching arXiv with query: {arxiv_query}")
            
            # Create search object
            search = arxiv.Search(
                query=arxiv_query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion(opts.sort_by.value),
                sort_order=arxiv.SortOrder(opts.sort_order.value)
            )
            
            # Execute search
            papers = []
            total_results = 0
            
            for result in search.results():
                total_results += 1
                paper = self._parse_paper(result)
                papers.append(paper)
                
                # Download PDF if requested
                if opts.download_pdfs:
                    pdf_path = os.path.join(
                        opts.pdf_dir,
                        f"{paper.arxiv_id}.pdf"
                    )
                    if not os.path.exists(pdf_path):
                        self.download_pdf(paper.arxiv_id, pdf_path)
            
            # Calculate search time
            search_time = time.time() - start_time
            
            self.logger.info(
                f"Found {len(papers)} papers in {search_time:.2f} seconds"
            )
            
            return SearchResult(
                success=True,
                query=arxiv_query,
                total_results=len(papers),
                papers=papers,
                search_time=search_time,
                metadata={
                    'max_results': max_results,
                    'sort_by': opts.sort_by.value,
                    'sort_order': opts.sort_order.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Search error: {str(e)}")
            return SearchResult(
                success=False,
                query=query,
                total_results=0,
                error_message=f"Search error: {str(e)}"
            )
    
    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """
        Get a specific paper by arXiv ID.
        
        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.08727")
            
        Returns:
            Paper object or None if not found
        """
        # Clean ID
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()
        
        result = self.search(query="", arxiv_id=arxiv_id, max_results=1)
        
        if result.success and result.papers:
            return result.papers[0]
        return None
    
    def download_pdf(
        self,
        arxiv_id: str,
        output_path: Optional[str] = None,
        chunk_size: int = 8192
    ) -> bool:
        """
        Download a paper's PDF.
        
        Args:
            arxiv_id: arXiv paper ID
            output_path: Where to save the PDF
            chunk_size: Download chunk size
            
        Returns:
            True if successful, False otherwise
        """
        # Clean ID
        arxiv_id = arxiv_id.replace('arXiv:', '').strip()
        
        # Default output path
        if not output_path:
            output_path = os.path.join(
                self.options.pdf_dir,
                f"{arxiv_id}.pdf"
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Construct PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            self.logger.info(f"Downloading PDF: {pdf_url}")
            
            response = requests.get(pdf_url, stream=True)
            response.raise_for_status()
            
            # Write PDF
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            
            self.logger.info(f"PDF saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Download error: {str(e)}")
            return False
    
    def search_by_date(
        self,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        categories: Optional[List[str]] = None,
        max_results: Optional[int] = None
    ) -> SearchResult:
        """
        Search papers by submission date.
        
        Args:
            start_date: Start date for search
            end_date: End date (default: now)
            categories: Filter by categories
            max_results: Maximum results
            
        Returns:
            SearchResult with papers from date range
        """
        if not end_date:
            end_date = datetime.now()
        
        # Date-based search is limited in arXiv API
        # We'll search broadly and filter
        query = "*"  # All papers
        
        options = SearchOptions(
            max_results=max_results or self.options.max_results,
            categories=categories or [],
            sort_by=SortCriterion.SUBMITTED,
            sort_order=SortOrder.DESCENDING
        )
        
        result = self.search(query, options=options)
        
        if result.success:
            # Filter by date
            filtered_papers = [
                paper for paper in result.papers
                if start_date <= paper.published <= end_date
            ]
            result.papers = filtered_papers
            result.total_results = len(filtered_papers)
        
        return result
    
    def get_citations(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get citation information for a paper.
        
        Note: arXiv doesn't provide citation data directly.
        This returns bibliographic information for citing the paper.
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Dictionary with citation information
        """
        paper = self.get_paper(arxiv_id)
        
        if not paper:
            return {"error": "Paper not found"}
        
        # Generate citation info
        authors_str = ", ".join([author.name for author in paper.authors])
        year = paper.published.year
        
        citation = {
            "bibtex": f"""@article{{{arxiv_id},
    title={{{paper.title}}},
    author={{{authors_str}}},
    journal={{arXiv preprint arXiv:{arxiv_id}}},
    year={{{year}}}
}}""",
            "mla": f"{authors_str}. \"{paper.title}.\" arXiv preprint arXiv:{arxiv_id} ({year}).",
            "apa": f"{authors_str} ({year}). {paper.title}. arXiv preprint arXiv:{arxiv_id}.",
            "chicago": f"{authors_str}. \"{paper.title}.\" arXiv preprint arXiv:{arxiv_id} ({year}).",
            "paper_info": paper.to_dict()
        }
        
        return citation


class ArxivSearchTool:
    """
    LLM Agent Tool wrapper for arXiv Search.
    
    This class provides a simplified interface designed for use in LLM agent
    frameworks like LangChain, AutoGPT, etc.
    """
    
    def __init__(self, default_options: Optional[SearchOptions] = None):
        """
        Initialize the tool.
        
        Args:
            default_options: Default search options
        """
        self.searcher = ArxivSearcher(default_options)
        self.name = "arxiv_search"
        self.description = (
            "Search for academic papers on arXiv.org. "
            "Input can be keywords, author names, or arXiv IDs. "
            "Returns paper metadata including title, authors, abstract, and PDF links."
        )
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run the tool with a search query.
        
        Args:
            query: Search query
            **kwargs: Additional options (author, title, categories, etc.)
            
        Returns:
            Dictionary with search results
        """
        # Parse kwargs
        result = self.searcher.search(
            query=query,
            title=kwargs.get('title'),
            author=kwargs.get('author'),
            abstract=kwargs.get('abstract'),
            categories=kwargs.get('categories'),
            max_results=kwargs.get('max_results', 10)
        )
        
        # Convert to dictionary for agent compatibility
        if result.success:
            papers_data = []
            for paper in result.papers:
                papers_data.append({
                    'arxiv_id': paper.arxiv_id,
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'abstract': paper.abstract[:500] + '...' if len(paper.abstract) > 500 else paper.abstract,
                    'published': paper.published.strftime('%Y-%m-%d'),
                    'categories': paper.categories,
                    'pdf_url': paper.pdf_url
                })
            
            return {
                'success': True,
                'total_results': result.total_results,
                'papers': papers_data,
                'query': result.query
            }
        else:
            return {
                'success': False,
                'error': result.error_message
            }
    
    async def arun(self, query: str, **kwargs) -> Dict[str, Any]:
        """Async version of run (currently just calls sync version)."""
        return self.run(query, **kwargs)
    
    def get_paper(self, arxiv_id: str) -> Dict[str, Any]:
        """
        Get a specific paper by ID.
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            Dictionary with paper information
        """
        paper = self.searcher.get_paper(arxiv_id)
        
        if paper:
            return {
                'success': True,
                'paper': paper.to_dict()
            }
        else:
            return {
                'success': False,
                'error': 'Paper not found'
            }
    
    def download_pdf(self, arxiv_id: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a paper's PDF.
        
        Args:
            arxiv_id: arXiv paper ID
            output_path: Where to save the PDF
            
        Returns:
            Dictionary with download status
        """
        success = self.searcher.download_pdf(arxiv_id, output_path)
        
        return {
            'success': success,
            'arxiv_id': arxiv_id,
            'output_path': output_path or f"./arxiv_papers/{arxiv_id}.pdf"
        }


@register_tool(
    tags=["research", "arxiv", "papers", "academic"]
)
def search_arxiv(
    query: str,
    author: Optional[str] = None,
    title: Optional[str] = None,
    categories: Optional[List[str]] = None,
    max_results: int = 10
) -> Dict[str, Any]:
    """Searches arXiv.org for academic papers.
    
    Args:
        query: General search query
        author: Search by author name
        title: Search in paper titles
        categories: Filter by arXiv categories (e.g., ["cs.AI", "cs.LG"])
        max_results: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    searcher = ArxivSearcher()
    result = searcher.search(
        query=query,
        author=author,
        title=title,
        categories=categories,
        max_results=max_results
    )
    
    if result.success:
        papers_data = []
        for paper in result.papers:
            papers_data.append({
                'arxiv_id': paper.arxiv_id,
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'abstract': paper.abstract[:300] + '...' if len(paper.abstract) > 300 else paper.abstract,
                'published': paper.published.strftime('%Y-%m-%d'),
                'pdf_url': paper.pdf_url
            })
        
        return {
            'success': True,
            'total_results': result.total_results,
            'papers': papers_data
        }
    else:
        return {
            'success': False,
            'error': result.error_message
        }


# Example usage and testing
if __name__ == "__main__":
    print("=== arXiv Search Tool Test ===\n")
    
    # Menu
    print("1. Search papers by keyword")
    print("2. Search papers by author")
    print("3. Get specific paper by ID")
    print("4. Search in specific category")
    print("5. Download paper PDF")
    print("6. Get citation information")
    
    choice = input("\nSelect option (1-6): ").strip()
    
    # Initialize searcher
    searcher = ArxivSearcher()
    
    if choice == "1":
        # Keyword search
        query = input("Enter search keywords: ").strip()
        max_results = input("Max results (default 10): ").strip()
        max_results = int(max_results) if max_results else 10
        
        print(f"\nSearching for: {query}")
        result = searcher.search(query, max_results=max_results)
        
        if result.success:
            print(f"\nFound {result.total_results} papers:")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join([a.name for a in paper.authors])}")
                print(f"   ID: {paper.arxiv_id}")
                print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
                print(f"   Categories: {', '.join(paper.categories)}")
                print(f"   Abstract: {paper.abstract[:200]}...")
        else:
            print(f"Error: {result.error_message}")
    
    elif choice == "2":
        # Author search
        author = input("Enter author name: ").strip()
        
        print(f"\nSearching for papers by: {author}")
        result = searcher.search("", author=author, max_results=10)
        
        if result.success:
            print(f"\nFound {result.total_results} papers by {author}:")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   ID: {paper.arxiv_id}")
                print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
        else:
            print(f"Error: {result.error_message}")
    
    elif choice == "3":
        # Get specific paper
        arxiv_id = input("Enter arXiv ID (e.g., 2301.08727): ").strip()
        
        print(f"\nFetching paper: {arxiv_id}")
        paper = searcher.get_paper(arxiv_id)
        
        if paper:
            print(f"\nTitle: {paper.title}")
            print(f"Authors: {', '.join([a.name for a in paper.authors])}")
            print(f"Published: {paper.published.strftime('%Y-%m-%d')}")
            print(f"Updated: {paper.updated.strftime('%Y-%m-%d')}")
            print(f"Categories: {', '.join(paper.categories)}")
            print(f"PDF URL: {paper.pdf_url}")
            print(f"\nAbstract:\n{paper.abstract}")
            
            if paper.comment:
                print(f"\nComment: {paper.comment}")
            if paper.journal_ref:
                print(f"Journal: {paper.journal_ref}")
        else:
            print("Paper not found")
    
    elif choice == "4":
        # Category search
        print("\nAvailable categories:")
        print("cs.AI - Artificial Intelligence")
        print("cs.LG - Machine Learning")
        print("cs.CV - Computer Vision")
        print("cs.CL - Computation and Language")
        print("math - Mathematics")
        print("physics - Physics")
        print("quant-ph - Quantum Physics")
        
        category = input("\nEnter category code: ").strip()
        
        print(f"\nSearching in category: {category}")
        result = searcher.search("", categories=[category], max_results=10)
        
        if result.success:
            print(f"\nFound {result.total_results} papers in {category}:")
            for i, paper in enumerate(result.papers, 1):
                print(f"\n{i}. {paper.title}")
                print(f"   Authors: {', '.join([a.name for a in paper.authors[:3]])}...")
                print(f"   ID: {paper.arxiv_id}")
        else:
            print(f"Error: {result.error_message}")
    
    elif choice == "5":
        # Download PDF
        arxiv_id = input("Enter arXiv ID to download: ").strip()
        
        print(f"\nDownloading PDF for: {arxiv_id}")
        success = searcher.download_pdf(arxiv_id)
        
        if success:
            print(f"PDF saved to: ./arxiv_papers/{arxiv_id}.pdf")
        else:
            print("Download failed")
    
    elif choice == "6":
        # Get citation
        arxiv_id = input("Enter arXiv ID for citation: ").strip()
        
        print(f"\nGenerating citations for: {arxiv_id}")
        citations = searcher.get_citations(arxiv_id)
        
        if "error" not in citations:
            print("\nBibTeX:")
            print(citations["bibtex"])
            print("\nAPA:")
            print(citations["apa"])
            print("\nMLA:")
            print(citations["mla"])
        else:
            print(f"Error: {citations['error']}")
    
    else:
        print("Invalid option")