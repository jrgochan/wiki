"""
Module for 3D visualization of the Wikipedia graph.
"""

import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
import pandas as pd
from .graph_utils import WikiGraph

class GraphVisualizer:
    """Class to handle 3D visualization of the WikiGraph."""
    
    def __init__(self, wiki_graph: WikiGraph):
        """
        Initialize the visualizer with a WikiGraph.
        
        Args:
            wiki_graph: WikiGraph instance to visualize
        """
        self.wiki_graph = wiki_graph
        self.layout_type = 'spring'  # Default layout
        self.node_positions = {}  # Will be populated when get_figure is called
        
        # Default visual attributes
        self.default_node_color = '#1f77b4'  # Default blue
        self.default_node_size = 10
        self.default_edge_color = 'rgba(211, 211, 211, 0.8)'  # Light gray with transparency
        self.default_edge_width = 1
        
        # Highlight colors for paths
        self.highlight_node_color = '#ff7f0e'  # Orange
        self.highlight_edge_color = 'rgba(255, 127, 14, 0.8)'  # Orange with transparency
        
        # Visual customization settings (can be modified by user)
        self.visual_settings = {
            'node_color': self.default_node_color,
            'node_opacity': 0.8,
            'node_size': self.default_node_size,
            'edge_color': self.default_edge_color,
            'edge_width': self.default_edge_width,
            'show_labels': False,
            'node_size_by_degree': False,
            'node_color_by_centrality': False,
        }
        
        # Filtering settings
        self.filters = {
            'min_degree': 0,
            'max_degree': None,
            'category': None,
            'search_term': None,
            'connected_component_only': False,
        }
        
        # Path highlighting
        self.highlighted_path = []
    
    def update_visual_settings(self, **kwargs) -> None:
        """
        Update visual customization settings.
        
        Args:
            **kwargs: Settings to update (e.g., node_color='red', edge_width=2)
        """
        for key, value in kwargs.items():
            if key in self.visual_settings:
                self.visual_settings[key] = value
    
    def update_filters(self, **kwargs) -> None:
        """
        Update filtering settings.
        
        Args:
            **kwargs: Filters to update (e.g., min_degree=2, category='Science')
        """
        for key, value in kwargs.items():
            if key in self.filters:
                self.filters[key] = value
    
    def highlight_path(self, source: Optional[str], target: Optional[str]) -> bool:
        """
        Highlight the shortest path between two nodes.
        
        Args:
            source: ID of the source node
            target: ID of the target node
            
        Returns:
            True if a path was found and highlighted, False otherwise
        """
        # Clear previous path
        self.highlighted_path = []
        
        if not source or not target:
            return False
        
        # Find shortest path
        path = self.wiki_graph.find_shortest_path(source, target)
        if not path:
            return False
        
        self.highlighted_path = path
        return True
    
    def clear_highlight(self) -> None:
        """Clear any highlighted path."""
        self.highlighted_path = []
    
    def _apply_filters(self) -> WikiGraph:
        """
        Apply current filters to the graph.
        
        Returns:
            Filtered WikiGraph
        """
        filtered_graph = self.wiki_graph
        
        # Filter by degree
        if self.filters['min_degree'] > 0 or self.filters['max_degree'] is not None:
            filtered_graph = filtered_graph.filter_by_degree(
                min_degree=self.filters['min_degree'],
                max_degree=self.filters['max_degree']
            )
        
        # Filter by category
        if self.filters['category']:
            filtered_graph = filtered_graph.filter_by_category(self.filters['category'])
        
        # Filter by largest connected component
        if self.filters['connected_component_only']:
            filtered_graph = filtered_graph.get_largest_connected_component()
        
        # Filter by search term in title
        if self.filters['search_term']:
            term = self.filters['search_term'].lower()
            
            def contains_term(node_id: str, attrs: Dict[str, Any]) -> bool:
                if node_id.lower().find(term) >= 0:
                    return True
                title = attrs.get('title', '')
                if isinstance(title, str) and title.lower().find(term) >= 0:
                    return True
                summary = attrs.get('summary', '')
                if isinstance(summary, str) and summary.lower().find(term) >= 0:
                    return True
                return False
            
            filtered_graph = filtered_graph.filter_nodes(contains_term)
        
        return filtered_graph
    
    def _get_trace_data(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Prepare data for Plotly visualization.
        
        Args:
            graph: NetworkX graph to visualize
            
        Returns:
            Dictionary with node and edge coordinates and attributes
        """
        # Get 3D positions for nodes
        self.node_positions = self.wiki_graph.get_layout(
            layout_type=self.layout_type, 
            seed=42
        )
        
        # Extract node positions as separate lists for plotting
        nodes = list(graph.nodes())
        
        # Node coordinates
        x_nodes = []
        y_nodes = []
        z_nodes = []
        
        # Node attributes for hover text
        titles = []
        urls = []
        summaries = []
        degrees = []
        node_sizes = []
        node_colors = []
        
        # Degree calculations for node sizes
        if self.visual_settings['node_size_by_degree']:
            degree_dict = dict(graph.degree())
            min_degree = min(degree_dict.values()) if degree_dict else 0
            max_degree = max(degree_dict.values()) if degree_dict else 1
            degree_range = max(1, max_degree - min_degree)  # Avoid division by zero
        
        # Centrality calculations for node colors
        if self.visual_settings['node_color_by_centrality']:
            centrality_dict = nx.degree_centrality(graph)
            min_cent = min(centrality_dict.values()) if centrality_dict else 0
            max_cent = max(centrality_dict.values()) if centrality_dict else 1
            cent_range = max(1e-6, max_cent - min_cent)  # Avoid division by zero
        
        # Prepare node data
        for node in nodes:
            # Skip nodes without positions (shouldn't happen, but just in case)
            if node not in self.node_positions:
                continue
            
            # Add coordinates
            x, y, z = self.node_positions[node]
            x_nodes.append(x)
            y_nodes.append(y)
            z_nodes.append(z)
            
            # Add attributes for hover text
            attrs = dict(graph.nodes[node])
            titles.append(attrs.get('title', node))
            urls.append(attrs.get('url', '#'))
            
            # Truncate summary for hover display
            summary = attrs.get('summary', '')
            if isinstance(summary, str) and len(summary) > 100:
                summary = summary[:97] + '...'
            summaries.append(summary)
            
            # Calculate degree
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            total_degree = in_degree + out_degree
            degrees.append(f"In: {in_degree}, Out: {out_degree}, Total: {total_degree}")
            
            # Determine node size based on settings
            if self.visual_settings['node_size_by_degree']:
                # Scale node size by degree
                size_factor = (total_degree - min_degree) / degree_range
                node_size = self.visual_settings['node_size'] * (0.5 + 2 * size_factor)
            else:
                node_size = self.visual_settings['node_size']
            
            # Adjust size for highlighted path nodes
            if node in self.highlighted_path:
                node_size *= 1.5
            
            node_sizes.append(node_size)
            
            # Determine node color based on settings
            if node in self.highlighted_path:
                # Highlighted path nodes get highlight color
                node_colors.append(self.highlight_node_color)
            elif self.visual_settings['node_color_by_centrality']:
                # Color by centrality - use a color scale from blue to red
                cent_value = centrality_dict.get(node, min_cent)
                color_val = (cent_value - min_cent) / cent_range
                
                # Generate a color in the blue-to-red spectrum
                r = int(255 * color_val)
                b = int(255 * (1 - color_val))
                g = int(100 + 50 * (0.5 - abs(0.5 - color_val)))
                
                node_colors.append(f'rgb({r},{g},{b})')
            else:
                # Default node color
                node_colors.append(self.visual_settings['node_color'])
        
        # Prepare edge data
        edges = list(graph.edges())
        
        # Lists to store edge endpoints
        x_edges = []
        y_edges = []
        z_edges = []
        edge_colors = []
        edge_widths = []
        edge_texts = []
        
        # Process each edge
        for u, v in edges:
            # Skip if either node is missing position data
            if u not in self.node_positions or v not in self.node_positions:
                continue
            
            # Get node positions
            x_u, y_u, z_u = self.node_positions[u]
            x_v, y_v, z_v = self.node_positions[v]
            
            # Add edge as a line segment with None separator
            x_edges.extend([x_u, x_v, None])
            y_edges.extend([y_u, y_v, None])
            z_edges.extend([z_u, z_v, None])
            
            # Determine if this edge is part of the highlighted path
            is_highlighted = (
                self.highlighted_path and 
                len(self.highlighted_path) > 1 and
                u in self.highlighted_path and 
                v in self.highlighted_path and
                abs(self.highlighted_path.index(u) - self.highlighted_path.index(v)) == 1
            )
            
            if is_highlighted:
                edge_colors.extend([self.highlight_edge_color] * 3)
                edge_widths.extend([self.visual_settings['edge_width'] * 2] * 3)
            else:
                edge_colors.extend([self.visual_settings['edge_color']] * 3)
                edge_widths.extend([self.visual_settings['edge_width']] * 3)
            
            # Text for hover
            edge_text = f"{u} → {v}"
            edge_texts.extend([edge_text, edge_text, None])
        
        # Calculate hover texts for nodes
        hover_texts = []
        for i in range(len(nodes)):
            hover_text = (
                f"<b>{titles[i]}</b><br>"
                f"Degrees: {degrees[i]}<br>"
                f"{summaries[i]}<br>"
                f"<a href='{urls[i]}' target='_blank'>Open in Wikipedia</a>"
            )
            hover_texts.append(hover_text)
        
        return {
            'nodes': {
                'x': x_nodes,
                'y': y_nodes,
                'z': z_nodes,
                'text': hover_texts,
                'titles': titles,
                'sizes': node_sizes,
                'colors': node_colors,
                'ids': nodes,
            },
            'edges': {
                'x': x_edges,
                'y': y_edges,
                'z': z_edges,
                'colors': edge_colors,
                'widths': edge_widths,
                'text': edge_texts,
            }
        }
    
    def get_figure(self) -> go.Figure:
        """
        Create a Plotly 3D graph figure based on current settings.
        
        Returns:
            Plotly Figure object
        """
        # Apply filters to get the graph to visualize
        filtered_graph = self._apply_filters()
        graph = filtered_graph.get_graph()
        
        # If we have a path to highlight, make sure the graph includes all path nodes
        if self.highlighted_path:
            # Add all nodes in the path if not already in the filtered graph
            for node in self.highlighted_path:
                if node not in graph:
                    graph.add_node(node, **self.wiki_graph.get_graph().nodes[node])
            
            # Add edges between consecutive nodes in the path
            for i in range(len(self.highlighted_path) - 1):
                u = self.highlighted_path[i]
                v = self.highlighted_path[i + 1]
                if not graph.has_edge(u, v):
                    # Get edge attributes from original graph if available
                    attrs = {}
                    if self.wiki_graph.get_graph().has_edge(u, v):
                        attrs = self.wiki_graph.get_graph().edges[u, v]
                    graph.add_edge(u, v, **attrs)
        
        # Prepare the data for plotting
        trace_data = self._get_trace_data(graph)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges as a scatter3d trace
        fig.add_trace(go.Scatter3d(
            x=trace_data['edges']['x'],
            y=trace_data['edges']['y'],
            z=trace_data['edges']['z'],
            mode='lines',
            line=dict(
                color=trace_data['edges']['colors'],
                width=self.visual_settings['edge_width'],  # Use a single width value instead of a list
            ),
            hoverinfo='text',
            text=trace_data['edges']['text'],
            name='Edges',
        ))
        
        # Add nodes as a scatter3d trace
        node_trace = go.Scatter3d(
            x=trace_data['nodes']['x'],
            y=trace_data['nodes']['y'],
            z=trace_data['nodes']['z'],
            mode='markers+text' if self.visual_settings['show_labels'] else 'markers',
            marker=dict(
                size=trace_data['nodes']['sizes'],
                color=trace_data['nodes']['colors'],
                opacity=self.visual_settings['node_opacity'],
                line=dict(width=0.5, color='rgb(50,50,50)'),
            ),
            text=trace_data['nodes']['titles'],
            hovertext=trace_data['nodes']['text'],
            hoverinfo='text',
            name='Nodes',
            customdata=trace_data['nodes']['ids'],  # Store node IDs for callbacks
        )
        
        # Add text settings if showing labels
        if self.visual_settings['show_labels']:
            node_trace.textposition = 'top center'
            node_trace.textfont = dict(size=10, color='black')
        
        fig.add_trace(node_trace)
        
        # Set up the layout
        title = "Wikipedia Articles Graph Visualization"
        
        # Add path information to title if path is highlighted
        if self.highlighted_path and len(self.highlighted_path) > 1:
            start = self.highlighted_path[0]
            end = self.highlighted_path[-1]
            title += f" (Path: {start} → {end}, {len(self.highlighted_path)-1} links)"
        
        # Add filter information to title
        filter_info = []
        if self.filters['min_degree'] > 0:
            filter_info.append(f"Min Degree: {self.filters['min_degree']}")
        if self.filters['max_degree'] is not None:
            filter_info.append(f"Max Degree: {self.filters['max_degree']}")
        if self.filters['category']:
            filter_info.append(f"Category: {self.filters['category']}")
        if self.filters['search_term']:
            filter_info.append(f"Search: '{self.filters['search_term']}'")
        if self.filters['connected_component_only']:
            filter_info.append("Largest Component Only")
        
        if filter_info:
            title += f"<br><sub>Filters: {', '.join(filter_info)}</sub>"
        
        # Set the layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title='', showticklabels=False, showgrid=True, zeroline=False),
                yaxis=dict(title='', showticklabels=False, showgrid=True, zeroline=False),
                zaxis=dict(title='', showticklabels=False, showgrid=True, zeroline=False),
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False,
            hovermode='closest',
        )
        
        return fig
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        # Apply filters to get statistics for the filtered graph
        filtered_graph = self._apply_filters()
        graph = filtered_graph.get_graph()
        original_graph = self.wiki_graph.get_graph()
        
        # Basic graph statistics
        stats = {
            'node_count': graph.number_of_nodes(),
            'edge_count': graph.number_of_edges(),
            'original_node_count': original_graph.number_of_nodes(),
            'original_edge_count': original_graph.number_of_edges(),
        }
        
        # Add path information if a path is highlighted
        if self.highlighted_path and len(self.highlighted_path) > 1:
            stats['path_source'] = self.highlighted_path[0]
            stats['path_target'] = self.highlighted_path[-1]
            stats['path_length'] = len(self.highlighted_path) - 1
            stats['path_nodes'] = self.highlighted_path
        
        return stats
    
    def get_node_table(self, limit: int = 100) -> pd.DataFrame:
        """
        Get a table of nodes and their properties.
        
        Args:
            limit: Maximum number of nodes to include
            
        Returns:
            Pandas DataFrame with node information
        """
        # Apply filters to get the filtered graph
        filtered_graph = self._apply_filters()
        graph = filtered_graph.get_graph()
        
        # Collect node data
        node_data = []
        for node in list(graph.nodes())[:limit]:
            attrs = dict(graph.nodes[node])
            in_degree = graph.in_degree(node)
            out_degree = graph.out_degree(node)
            
            node_data.append({
                'node_id': node,
                'title': attrs.get('title', node),
                'in_degree': in_degree,
                'out_degree': out_degree,
                'total_degree': in_degree + out_degree,
                'url': attrs.get('url', ''),
                'categories': ', '.join(attrs.get('categories', [])),
                'in_path': node in self.highlighted_path,
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(node_data)
        
        # Sort by total degree by default
        if not df.empty:
            df = df.sort_values('total_degree', ascending=False)
        
        return df
    
    def get_layout_options(self) -> List[str]:
        """
        Get available layout options.
        
        Returns:
            List of layout type strings
        """
        return list(self.wiki_graph._layouts.keys())
    
    def set_layout(self, layout_type: str) -> None:
        """
        Set the layout type.
        
        Args:
            layout_type: Layout type to use
        """
        if layout_type in self.wiki_graph._layouts:
            self.layout_type = layout_type
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")
    
    def get_categories(self) -> List[str]:
        """
        Get all unique categories in the graph.
        
        Returns:
            List of category strings
        """
        categories = set()
        for _, attrs in self.wiki_graph.get_graph().nodes(data=True):
            node_categories = attrs.get('categories', [])
            if isinstance(node_categories, list):
                categories.update(node_categories)
        
        return sorted(list(categories))
