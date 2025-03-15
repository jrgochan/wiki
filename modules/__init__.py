"""
Modules package for the Wikipedia 3D Graph Visualization tool.
"""

# Import main classes for easier access from modules package
from .wiki_api import WikipediaDataFetcher
from .graph_utils import WikiGraph
from .visualization import GraphVisualizer

__all__ = ['WikipediaDataFetcher', 'WikiGraph', 'GraphVisualizer']
