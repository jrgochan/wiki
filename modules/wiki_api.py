"""
Module for interacting with the Wikipedia API and retrieving article data.
"""

import requests
import wikipediaapi
import networkx as nx
import time
from tqdm import tqdm
import random
import json
import os
from typing import Dict, List, Set, Tuple, Optional, Any

class WikipediaDataFetcher:
    """Class to handle data acquisition from Wikipedia API."""
    
    def __init__(self, language: str = "en"):
        """
        Initialize the Wikipedia API client.
        
        Args:
            language: Language code for Wikipedia (default: "en" for English)
        """
        self.wiki = wikipediaapi.Wikipedia(
            language=language,
            user_agent="WikiGraphVisualizer/1.0 (https://github.com/jrgochan/wiki)"
        )
        self.session = requests.Session()
        # Base URL for the MediaWiki API
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_article(self, title: str) -> Optional[wikipediaapi.WikipediaPage]:
        """
        Get a Wikipedia article by title.
        
        Args:
            title: Title of the Wikipedia article
            
        Returns:
            WikipediaPage object or None if article doesn't exist
        """
        page = self.wiki.page(title)
        if not page.exists():
            print(f"Article '{title}' does not exist on Wikipedia")
            return None
        return page
    
    def get_article_links(self, title: str) -> Dict[str, Set[str]]:
        """
        Get all links from a Wikipedia article.
        
        Args:
            title: Title of the Wikipedia article
            
        Returns:
            Dictionary with 'links' and 'backlinks' keys containing sets of article titles
        """
        page = self.get_article(title)
        if not page:
            return {"links": set(), "backlinks": set()}
        
        # Get all links
        links = {link_title for link_title in page.links.keys()}
        
        # Get backlinks (this can be slow for popular articles)
        backlinks = set()
        # For demonstration, we'll limit backlinks to avoid API throttling
        backlinks_limit = 100
        
        # Get backlinks using the MediaWiki API directly
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
            backlinks = {item["title"] for item in data["query"]["backlinks"]}
        
        return {"links": links, "backlinks": backlinks}
    
    def build_graph_from_seed(
        self, 
        seed_articles: List[str], 
        max_articles: int = 50, 
        depth: int = 2,
        save_to_file: bool = True
    ) -> nx.DiGraph:
        """
        Build a directed graph starting from seed articles and following links up to a certain depth.
        
        Args:
            seed_articles: List of article titles to start from
            max_articles: Maximum number of articles to include
            depth: How many levels of links to follow
            save_to_file: Whether to save the resulting graph to a file
            
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        visited = set()
        to_visit = [(article, 0) for article in seed_articles]  # (article_title, depth)
        
        print(f"Building graph from {len(seed_articles)} seed articles with max depth {depth} and max size {max_articles}")
        
        with tqdm(total=max_articles) as pbar:
            while to_visit and len(visited) < max_articles:
                # Get next article to process
                current_article, current_depth = to_visit.pop(0)
                
                # Skip if already visited
                if current_article in visited:
                    continue
                
                # Try to get article data
                try:
                    page = self.get_article(current_article)
                    if not page:
                        continue
                    
                    # Mark as visited
                    visited.add(current_article)
                    pbar.update(1)
                    
                    # Add node with some metadata
                    G.add_node(
                        current_article,
                        title=current_article,
                        url=page.fullurl,
                        summary=page.summary[0:200] if page.summary else "",
                        categories=[cat for cat in page.categories.keys()],
                    )
                    
                    # If we've reached max depth, don't get links
                    if current_depth >= depth:
                        continue
                    
                    # Get links from the article
                    links = page.links
                    
                    # Limit links to a reasonable number to avoid huge graphs
                    link_sample = list(links.keys())
                    random.shuffle(link_sample)
                    link_sample = link_sample[:min(20, len(link_sample))]
                    
                    # Add links to the graph and queue them for processing
                    for link_title in link_sample:
                        # Ignore non-article links (categories, files, etc.)
                        if any(ns in link_title for ns in ["Category:", "File:", "Template:", "Wikipedia:", "Help:", "Portal:"]):
                            continue
                        
                        # Add edge to the graph
                        G.add_edge(current_article, link_title)
                        
                        # Add to visit queue if not already visited
                        if link_title not in visited:
                            to_visit.append((link_title, current_depth + 1))
                    
                    # Be nice to the Wikipedia API - add small delay
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error processing article '{current_article}': {e}")
                    continue
        
        print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Save graph to file if requested
        if save_to_file:
            self.save_graph_to_file(G, "wiki_graph")
        
        return G
    
    def save_graph_to_file(self, G: nx.DiGraph, filename_base: str) -> str:
        """
        Save a graph to files in JSON format for nodes and edges.
        
        Args:
            G: NetworkX graph to save
            filename_base: Base name for the output files
            
        Returns:
            Path to the saved node file
        """
        # Extract node data
        nodes_data = []
        for node in G.nodes():
            node_data = {
                "id": node,
                "title": node,
                **{k: v for k, v in G.nodes[node].items()}
            }
            nodes_data.append(node_data)
        
        # Extract edge data
        edges_data = []
        for src, dst in G.edges():
            edge_data = {
                "source": src,
                "target": dst,
                **{k: v for k, v in G.edges[src, dst].items()}
            }
            edges_data.append(edge_data)
        
        # Save to JSON files
        nodes_file = os.path.join(self.data_dir, f"{filename_base}_nodes.json")
        edges_file = os.path.join(self.data_dir, f"{filename_base}_edges.json")
        
        with open(nodes_file, "w") as f:
            json.dump(nodes_data, f, indent=2)
        
        with open(edges_file, "w") as f:
            json.dump(edges_data, f, indent=2)
        
        print(f"Graph saved to {nodes_file} and {edges_file}")
        return nodes_file
    
    def load_graph_from_file(self, filename_base: str) -> nx.DiGraph:
        """
        Load a graph from JSON files.
        
        Args:
            filename_base: Base name for the input files
            
        Returns:
            NetworkX directed graph
        """
        nodes_file = os.path.join(self.data_dir, f"{filename_base}_nodes.json")
        edges_file = os.path.join(self.data_dir, f"{filename_base}_edges.json")
        
        # Check if files exist
        if not os.path.exists(nodes_file) or not os.path.exists(edges_file):
            raise FileNotFoundError(f"Graph files {filename_base}_nodes.json and/or {filename_base}_edges.json not found")
        
        # Load data from files
        with open(nodes_file, "r") as f:
            nodes_data = json.load(f)
        
        with open(edges_file, "r") as f:
            edges_data = json.load(f)
        
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node_data in nodes_data:
            node_id = node_data.pop("id")
            G.add_node(node_id, **node_data)
        
        # Add edges with attributes
        for edge_data in edges_data:
            src = edge_data.pop("source")
            dst = edge_data.pop("target")
            G.add_edge(src, dst, **edge_data)
        
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def generate_sample_dataset(
        self, 
        size: int = 100, 
        seed_articles: Optional[List[str]] = None
    ) -> nx.DiGraph:
        """
        Generate a sample dataset for demonstration purposes.
        If Wikipedia API is not available, this creates a synthetic graph.
        
        Args:
            size: Number of nodes in the sample dataset
            seed_articles: Optional list of seed articles to start from
            
        Returns:
            NetworkX directed graph
        """
        if seed_articles:
            try:
                # Try to build a real graph from Wikipedia
                return self.build_graph_from_seed(
                    seed_articles=seed_articles,
                    max_articles=size,
                    depth=2,
                    save_to_file=True
                )
            except Exception as e:
                print(f"Error building real graph: {e}")
                print("Falling back to synthetic dataset")
        
        # Generate a synthetic dataset
        print(f"Generating synthetic Wikipedia graph with {size} nodes")
        G = nx.scale_free_graph(size)
        
        # Convert to DiGraph if not already
        if not isinstance(G, nx.DiGraph):
            G = nx.DiGraph(G)
        
        # Add fake Wikipedia article metadata
        topics = [
            "History", "Science", "Technology", "Art", "Literature", 
            "Music", "Film", "Politics", "Sports", "Geography"
        ]
        
        for i, node in enumerate(G.nodes()):
            topic = random.choice(topics)
            G.nodes[node]['id'] = f"article_{i}"
            G.nodes[node]['title'] = f"{topic} article {i}"
            G.nodes[node]['url'] = f"https://en.wikipedia.org/wiki/{topic}_{i}"
            G.nodes[node]['summary'] = f"This is a sample article about {topic.lower()}."
            G.nodes[node]['categories'] = [topic, random.choice(topics)]
        
        # Save the synthetic dataset
        self.save_graph_to_file(G, "wiki_graph_synthetic")
        
        return G
    
    def search_wikipedia(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search Wikipedia for articles matching the query.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            
        Returns:
            List of dictionaries with article data (title, snippet, etc.)
        """
        if not query or len(query.strip()) == 0:
            return []
            
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srinfo": "snippet",
            "srprop": "snippet|titlesnippet|sectiontitle",
        }
        
        try:
            response = self.session.get(self.api_url, params=params)
            data = response.json()
            
            results = []
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    # Clean up the snippet (remove HTML tags)
                    snippet = item.get("snippet", "")
                    snippet = snippet.replace("<span class=\"searchmatch\">", "")
                    snippet = snippet.replace("</span>", "")
                    
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": snippet,
                        "pageid": item.get("pageid", 0),
                    })
            
            return results
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
            return []
