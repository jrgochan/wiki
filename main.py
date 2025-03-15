#!/usr/bin/env python3
"""
Main entry point for the Wikipedia 3D Graph Visualization tool.
This script provides a command-line interface to run the application.
"""

import argparse
import os
import sys
from modules.wiki_api import WikipediaDataFetcher
from modules.graph_utils import WikiGraph
from modules.visualization import GraphVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Wikipedia 3D Graph Visualization Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--port', type=int, default=8050,
        help='Port to run the web server on'
    )
    parser.add_argument(
        '--host', type=str, default='0.0.0.0',
        help='Host to run the web server on'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run in debug mode'
    )
    parser.add_argument(
        '--generate-sample', action='store_true',
        help='Generate a sample dataset before starting the server'
    )
    parser.add_argument(
        '--sample-size', type=int, default=100,
        help='Size of the sample dataset to generate'
    )
    parser.add_argument(
        '--seed-articles', type=str, nargs='+',
        help='Seed articles to use for generating the graph. If not provided, a synthetic graph will be generated.'
    )
    
    return parser.parse_args()

def generate_sample_data(args):
    """Generate a sample dataset."""
    print("Generating sample dataset...")
    wiki_fetcher = WikipediaDataFetcher()
    
    if args.seed_articles:
        print(f"Using seed articles: {', '.join(args.seed_articles)}")
        try:
            graph = wiki_fetcher.build_graph_from_seed(
                seed_articles=args.seed_articles,
                max_articles=args.sample_size,
                depth=2,
                save_to_file=True
            )
            print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error building graph from seeds: {e}")
            print("Falling back to synthetic dataset...")
            graph = wiki_fetcher.generate_sample_dataset(size=args.sample_size)
            print(f"Generated synthetic graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    else:
        print("No seed articles provided, generating synthetic dataset...")
        graph = wiki_fetcher.generate_sample_dataset(size=args.sample_size)
        print(f"Generated synthetic graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    print("Sample dataset generation complete.")

def run_web_app(args):
    """Run the web application."""
    # Import here to avoid loading unnecessary modules if only generating sample data
    from app import app
    
    print(f"Starting web server on http://{args.host}:{args.port}...")
    app.run_server(debug=args.debug, host=args.host, port=args.port)

def main():
    """Main entry point."""
    args = parse_args()
    
    # Generate sample data if requested
    if args.generate_sample:
        generate_sample_data(args)
    
    # Run the web application
    run_web_app(args)

if __name__ == "__main__":
    main()
