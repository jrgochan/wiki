#!/usr/bin/env python3
"""
Utility script to generate a sample dataset for the Wikipedia 3D Graph Visualization tool.
This can be used to quickly create test data without using the web interface.
"""

import argparse
import os
import sys
import networkx as nx
from modules.wiki_api import WikipediaDataFetcher
from modules.graph_utils import WikiGraph

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate a sample dataset for the Wikipedia 3D Graph Visualization tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add arguments
    parser.add_argument(
        '--size', type=int, default=100,
        help='Number of nodes in the sample dataset'
    )
    parser.add_argument(
        '--seed-articles', type=str, nargs='+',
        help='Seed articles to use for generating the graph. If not provided, a synthetic graph will be generated.'
    )
    parser.add_argument(
        '--depth', type=int, default=2,
        help='Depth of crawl from seed articles'
    )
    parser.add_argument(
        '--output', type=str, default='wiki_graph_sample',
        help='Base filename for the output (without extension)'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Force generation of a synthetic dataset even if seed articles are provided'
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    print("Generating sample dataset...")
    wiki_fetcher = WikipediaDataFetcher()
    
    try:
        if args.seed_articles and not args.synthetic:
            print(f"Using seed articles: {', '.join(args.seed_articles)}")
            try:
                graph = wiki_fetcher.build_graph_from_seed(
                    seed_articles=args.seed_articles,
                    max_articles=args.size,
                    depth=args.depth,
                    save_to_file=False  # We'll save it with our own filename
                )
                print(f"Successfully built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            except Exception as e:
                print(f"Error building graph from seeds: {e}")
                print("Falling back to synthetic dataset...")
                graph = wiki_fetcher.generate_sample_dataset(size=args.size)
                print(f"Generated synthetic graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        else:
            if args.synthetic:
                print("Synthetic dataset requested.")
            else:
                print("No seed articles provided.")
            
            print(f"Generating synthetic dataset with {args.size} nodes...")
            graph = wiki_fetcher.generate_sample_dataset(size=args.size)
            print(f"Generated synthetic graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        
        # Save the graph with the specified output filename
        output_path = wiki_fetcher.save_graph_to_file(graph, args.output)
        print(f"Saved graph to {output_path}")
        
        # Print some statistics about the graph
        print("\nGraph Statistics:")
        print(f"  Nodes: {graph.number_of_nodes()}")
        print(f"  Edges: {graph.number_of_edges()}")
        
        # Compute some additional metrics
        if graph.number_of_nodes() > 0:
            # Convert to undirected for some metrics
            undirected = graph.to_undirected()
            try:
                largest_cc = max(nx.connected_components(undirected), key=len)
                print(f"  Largest connected component: {len(largest_cc)} nodes")
            except:
                print("  Could not compute connected components.")
            
            try:
                # Get degree statistics
                degrees = [d for _, d in graph.degree()]
                avg_degree = sum(degrees) / len(degrees) if degrees else 0
                max_degree = max(degrees) if degrees else 0
                print(f"  Average degree: {avg_degree:.2f}")
                print(f"  Maximum degree: {max_degree}")
            except:
                print("  Could not compute degree statistics.")
            
            try:
                # Get some high-degree nodes as examples
                high_degree_nodes = sorted(graph.degree(), key=lambda x: x[1], reverse=True)[:5]
                print("\nSome central nodes:")
                for node, degree in high_degree_nodes:
                    node_attrs = graph.nodes[node]
                    title = node_attrs.get('title', node)
                    print(f"  {title} (degree: {degree})")
            except:
                print("  Could not identify central nodes.")
        
        print("\nYou can now run the application to visualize this dataset:")
        print(f"  python main.py")
        print("Then click 'Use Sample Dataset' in the web interface.")
        
    except Exception as e:
        print(f"Error generating sample dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
