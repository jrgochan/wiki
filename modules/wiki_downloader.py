"""
Module for multi-threaded Wikipedia article downloading.
"""

import os
import json
import time
import threading
import queue
import logging
import random
from typing import Dict, List, Set, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import wikipediaapi
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wiki_download.log")
    ]
)
logger = logging.getLogger("WikiDownloader")

class WikiDownloader:
    """Class to handle multi-threaded downloading of Wikipedia articles."""
    
    def __init__(self, 
                base_dir: str = "data/wiki_dump", 
                language: str = "en",
                max_threads: int = 10,
                max_articles_per_category: int = 100,
                articles_per_batch: int = 20):
        """
        Initialize the downloader.
        
        Args:
            base_dir: Base directory to store downloaded articles
            language: Wikipedia language code
            max_threads: Maximum number of download threads
            max_articles_per_category: Maximum articles to download per category
            articles_per_batch: Number of articles to process in each batch
        """
        self.base_dir = base_dir
        self.language = language
        self.max_threads = max_threads
        self.max_articles_per_category = max_articles_per_category
        self.articles_per_batch = articles_per_batch
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="WikiGraphVisualizer/1.0 (https://github.com/yourusername/wiki-graph-viz)"
        )
        
        # MediaWiki API session
        self.session = requests.Session()
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Track downloaded articles to avoid duplicates
        self.downloaded_articles = set()
        self.article_queue = queue.Queue()
        self.category_queue = queue.Queue()
        
        # Progress tracking
        self.total_articles = 0
        self.completed_articles = 0
        self.failed_articles = 0
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "articles_downloaded": 0,
            "categories_processed": 0,
            "total_size_kb": 0,
            "start_time": 0,
            "end_time": 0,
        }

    def get_top_level_categories(self) -> List[str]:
        """
        Get a list of top-level Wikipedia categories.
        
        Returns:
            List of category names
        """
        # Main top-level categories in Wikipedia
        top_categories = [
            "Category:Arts",
            "Category:Culture",
            "Category:Geography",
            "Category:Health",
            "Category:History",
            "Category:Mathematics",
            "Category:Nature",
            "Category:People",
            "Category:Philosophy",
            "Category:Religion",
            "Category:Science",
            "Category:Society",
            "Category:Sports",
            "Category:Technology",
            "Category:All categories",
        ]
        
        random.shuffle(top_categories)  # Randomize to distribute load
        return top_categories
    
    def get_subcategories(self, category_title: str) -> List[str]:
        """
        Get subcategories for a given category.
        
        Args:
            category_title: Category title
            
        Returns:
            List of subcategory names
        """
        if not category_title.startswith("Category:"):
            category_title = f"Category:{category_title}"
        
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "subcat",
            "cmlimit": 50,
        }
        
        try:
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            subcategories = []
            if "query" in data and "categorymembers" in data["query"]:
                for member in data["query"]["categorymembers"]:
                    subcategories.append(member["title"])
            
            return subcategories
        except Exception as e:
            logger.error(f"Error getting subcategories for {category_title}: {e}")
            return []
    
    def get_category_articles(self, category_title: str, max_articles: int = 50) -> List[str]:
        """
        Get articles in a category.
        
        Args:
            category_title: Category title
            max_articles: Maximum number of articles to retrieve
            
        Returns:
            List of article titles
        """
        if not category_title.startswith("Category:"):
            category_title = f"Category:{category_title}"
        
        params = {
            "action": "query",
            "format": "json",
            "list": "categorymembers",
            "cmtitle": category_title,
            "cmtype": "page",
            "cmlimit": max_articles,
        }
        
        try:
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            articles = []
            if "query" in data and "categorymembers" in data["query"]:
                for member in data["query"]["categorymembers"]:
                    # Only include main namespace articles
                    if ":" not in member["title"]:
                        articles.append(member["title"])
            
            return articles
        except Exception as e:
            logger.error(f"Error getting articles for {category_title}: {e}")
            return []
    
    def download_article(self, title: str) -> bool:
        """
        Download a single Wikipedia article.
        
        Args:
            title: Article title
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if already downloaded
            if title in self.downloaded_articles:
                return True
            
            # Get article
            page = self.wiki.page(title)
            if not page.exists():
                logger.warning(f"Article '{title}' does not exist")
                return False
            
            # Create category directory
            category_path = os.path.join(self.base_dir, "articles")
            os.makedirs(category_path, exist_ok=True)
            
            # First character of title for subdirectory (to distribute files)
            first_char = title[0].upper()
            if not first_char.isalnum():
                first_char = "0"  # Special characters go in "0" directory
            
            char_path = os.path.join(category_path, first_char)
            os.makedirs(char_path, exist_ok=True)
            
            # Create safe filename
            safe_title = "".join(c if c.isalnum() else "_" for c in title)
            file_path = os.path.join(char_path, f"{safe_title}.json")
            
            # Extract article data
            article_data = {
                "title": page.title,
                "summary": page.summary,
                "content": page.text,
                "url": page.fullurl,
                "categories": list(page.categories.keys()),
                "links": list(page.links.keys()),
                "backlinks": [],  # Could be populated later
                "last_modified": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Get a sample of backlinks (this is slow for popular articles)
            # So we just get a small sample for demonstration
            backlinks_limit = 20
            params = {
                "action": "query",
                "format": "json",
                "list": "backlinks",
                "bltitle": title,
                "bllimit": backlinks_limit,
            }
            
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            if "query" in data and "backlinks" in data["query"]:
                article_data["backlinks"] = [item["title"] for item in data["query"]["backlinks"]]
            
            # Save article to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(article_data, f, indent=2, ensure_ascii=False)
            
            # Track statistics
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            with self.lock:
                self.downloaded_articles.add(title)
                self.completed_articles += 1
                self.stats["articles_downloaded"] += 1
                self.stats["total_size_kb"] += file_size
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading article '{title}': {e}")
            with self.lock:
                self.failed_articles += 1
            return False
    
    def article_worker(self):
        """Worker function for downloading articles from the queue."""
        while not self.stop_event.is_set():
            try:
                article = self.article_queue.get(timeout=0.5)
                self.download_article(article)
                self.article_queue.task_done()
                
                # Be nice to the Wikipedia servers - add a small random delay
                time.sleep(random.uniform(0.1, 0.5))
                
            except queue.Empty:
                # Queue is empty, wait for more items
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in article worker: {e}")
                self.article_queue.task_done()
    
    def category_worker(self):
        """Worker function for processing categories from the queue."""
        while not self.stop_event.is_set():
            try:
                category = self.category_queue.get(timeout=0.5)
                
                # Get articles in this category
                articles = self.get_category_articles(
                    category, 
                    max_articles=self.max_articles_per_category
                )
                
                # Add articles to the download queue
                for article in articles:
                    if article not in self.downloaded_articles:
                        self.article_queue.put(article)
                        with self.lock:
                            self.total_articles += 1
                
                # Get subcategories
                subcategories = self.get_subcategories(category)
                
                # Add subcategories to the queue (breadth-first exploration)
                for subcat in subcategories[:10]:  # Limit to avoid explosion
                    self.category_queue.put(subcat)
                
                with self.lock:
                    self.stats["categories_processed"] += 1
                
                self.category_queue.task_done()
                
                # Be nice to the Wikipedia servers - add a small delay
                time.sleep(random.uniform(0.2, 1.0))
                
            except queue.Empty:
                # Queue is empty, wait for more items
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Error in category worker: {e}")
                self.category_queue.task_done()
    
    def start_download(self, 
                      max_downloads: int = 1000, 
                      initial_categories: Optional[List[str]] = None,
                      initial_articles: Optional[List[str]] = None) -> None:
        """
        Start downloading Wikipedia articles in multiple threads.
        
        Args:
            max_downloads: Maximum number of articles to download
            initial_categories: Initial categories to start with (if None, uses top-level categories)
            initial_articles: Initial articles to start with (optional)
        """
        # Reset state
        self.stop_event.clear()
        self.downloaded_articles.clear()
        self.total_articles = 0
        self.completed_articles = 0
        self.failed_articles = 0
        
        # Clear queues
        while not self.article_queue.empty():
            self.article_queue.get()
            self.article_queue.task_done()
        
        while not self.category_queue.empty():
            self.category_queue.get()
            self.category_queue.task_done()
        
        # Initialize stats
        self.stats = {
            "articles_downloaded": 0,
            "categories_processed": 0,
            "total_size_kb": 0,
            "start_time": time.time(),
            "end_time": 0,
        }
        
        # Add initial articles to queue
        if initial_articles:
            for article in initial_articles:
                self.article_queue.put(article)
                self.total_articles += 1
        
        # Add initial categories to queue
        if initial_categories:
            for category in initial_categories:
                self.category_queue.put(category)
        else:
            # Start with top-level categories
            for category in self.get_top_level_categories():
                self.category_queue.put(category)
        
        # Create and start category worker threads
        category_threads = []
        for _ in range(2):  # Use fewer threads for category processing
            thread = threading.Thread(target=self.category_worker)
            thread.daemon = True
            thread.start()
            category_threads.append(thread)
        
        # Create and start article worker threads
        article_threads = []
        for _ in range(self.max_threads):
            thread = threading.Thread(target=self.article_worker)
            thread.daemon = True
            thread.start()
            article_threads.append(thread)
        
        # Monitor progress in the main thread
        try:
            with tqdm(total=max_downloads, desc="Downloading articles") as pbar:
                while (self.completed_articles < max_downloads and 
                       not self.stop_event.is_set()):
                    
                    # Update progress bar
                    pbar.n = self.completed_articles
                    pbar.refresh()
                    
                    # Check if we have enough articles in queue
                    if (self.total_articles < max_downloads and 
                        self.article_queue.qsize() < self.articles_per_batch):
                        # We need more articles, but we'll let the category workers find them
                        pass
                    
                    # Slow down the main thread
                    time.sleep(0.5)
                    
                    # Break if all work is done
                    if (self.article_queue.empty() and self.category_queue.empty() and 
                        self.completed_articles >= self.total_articles):
                        break
            
            # Signal threads to stop
            self.stop_event.set()
            
            # Final update
            self.stats["end_time"] = time.time()
            
            # Print summary
            duration = self.stats["end_time"] - self.stats["start_time"]
            logger.info(f"Download summary:")
            logger.info(f"  Articles downloaded: {self.stats['articles_downloaded']}")
            logger.info(f"  Categories processed: {self.stats['categories_processed']}")
            logger.info(f"  Total size: {self.stats['total_size_kb'] / 1024:.2f} MB")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Download rate: {self.stats['articles_downloaded'] / duration:.2f} articles/second")
            
        except KeyboardInterrupt:
            logger.info("Download interrupted by user")
            self.stop_event.set()
            
        # Wait for threads to finish
        for thread in article_threads + category_threads:
            thread.join(timeout=1.0)

    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get current download statistics.
        
        Returns:
            Dictionary with download statistics
        """
        stats = dict(self.stats)
        
        # Add current progress
        stats.update({
            "total_articles": self.total_articles,
            "completed_articles": self.completed_articles,
            "failed_articles": self.failed_articles,
            "queued_articles": self.article_queue.qsize(),
            "queued_categories": self.category_queue.qsize(),
            "running": not self.stop_event.is_set(),
        })
        
        # Calculate elapsed time
        if stats["start_time"] > 0:
            current_time = time.time() if stats["end_time"] == 0 else stats["end_time"]
            stats["elapsed_time"] = current_time - stats["start_time"]
        else:
            stats["elapsed_time"] = 0
        
        return stats
    
    def stop_download(self) -> None:
        """Stop the download process."""
        self.stop_event.set()
        logger.info("Download stopping (may take a moment to complete)...")
        
        # Wait for queues to be empty
        self.article_queue.join()
        self.category_queue.join()
        
        self.stats["end_time"] = time.time()
        logger.info("Download stopped")
    
    def is_downloading(self) -> bool:
        """Check if download is in progress."""
        return not self.stop_event.is_set()
