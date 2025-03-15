"""
Module for graph construction, analysis, and manipulation.
"""

import networkx as nx
import numpy as np
import random
from typing import Dict, List, Set, Tuple, Optional, Any, Callable

class WikiGraph:
    """Class to handle graph operations for the Wikipedia article network."""
    
    def __init__(self, graph: Optional[nx.DiGraph] = None):
        """
        Initialize the WikiGraph with an optional existing graph.
        
        Args:
            graph: Existing NetworkX graph (optional)
        """
        self.graph = graph if graph is not None else nx.DiGraph()
        self._node_positions = {}  # Cache for node positions in layouts
        self._layouts = {
            'spring': self._spring_layout,
            'circular': self._circular_layout,
            'kamada_kawai': self._kamada_kawai_layout,
            'spectral': self._spectral_layout,
            'random': self._random_layout,
            'shell': self._shell_layout,
        }
        
    def get_graph(self) -> nx.DiGraph:
        """
        Get the underlying NetworkX graph.
        
        Returns:
            The NetworkX graph object
        """
        return self.graph
    
    def set_graph(self, graph: nx.DiGraph) -> None:
        """
        Set the graph to a new NetworkX graph.
        
        Args:
            graph: NetworkX graph to set
        """
        self.graph = graph
        # Clear cached positions when graph changes
        self._node_positions = {}
    
    def get_node_count(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            Number of nodes
        """
        return self.graph.number_of_nodes()
    
    def get_edge_count(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            Number of edges
        """
        return self.graph.number_of_edges()
    
    def get_node_degrees(self) -> Dict[str, Dict[str, int]]:
        """
        Get in and out degrees for all nodes.
        
        Returns:
            Dictionary mapping node ids to their in and out degrees
        """
        result = {}
        for node in self.graph.nodes():
            result[node] = {
                'in_degree': self.graph.in_degree(node),
                'out_degree': self.graph.out_degree(node),
                'total_degree': self.graph.in_degree(node) + self.graph.out_degree(node)
            }
        return result
    
    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """
        Get all attributes for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary of node attributes
        """
        if node_id not in self.graph.nodes():
            raise ValueError(f"Node '{node_id}' not found in graph")
        
        return dict(self.graph.nodes[node_id])
    
    def get_all_node_attributes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get attributes for all nodes.
        
        Returns:
            Dictionary mapping node ids to their attributes
        """
        return {node: dict(attrs) for node, attrs in self.graph.nodes(data=True)}
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find the shortest path between two nodes.
        
        Args:
            source: ID of the source node
            target: ID of the target node
            
        Returns:
            List of node IDs in the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return None
    
    def get_path_length(self, source: str, target: str) -> Optional[int]:
        """
        Get the length of the shortest path between two nodes.
        
        Args:
            source: ID of the source node
            target: ID of the target node
            
        Returns:
            Length of the shortest path, or None if no path exists
        """
        try:
            return nx.shortest_path_length(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return None
    
    def compute_centrality(self, method: str = 'degree') -> Dict[str, float]:
        """
        Compute centrality measures for nodes.
        
        Args:
            method: Centrality method ('degree', 'betweenness', 'closeness', 'eigenvector')
            
        Returns:
            Dictionary mapping node ids to centrality values
        """
        if method == 'degree':
            return nx.degree_centrality(self.graph)
        elif method == 'betweenness':
            return nx.betweenness_centrality(self.graph)
        elif method == 'closeness':
            return nx.closeness_centrality(self.graph)
        elif method == 'eigenvector':
            return nx.eigenvector_centrality(self.graph, max_iter=100, tol=1e-6)
        else:
            raise ValueError(f"Unknown centrality method: {method}")
    
    def filter_nodes(self, 
                    condition: Callable[[str, Dict[str, Any]], bool]) -> 'WikiGraph':
        """
        Filter nodes based on a condition.
        
        Args:
            condition: Function that takes node ID and attributes and returns True/False
            
        Returns:
            New WikiGraph with filtered nodes
        """
        filtered_nodes = [
            node for node in self.graph.nodes() 
            if condition(node, dict(self.graph.nodes[node]))
        ]
        
        # Create subgraph with the filtered nodes
        subgraph = self.graph.subgraph(filtered_nodes).copy()
        
        return WikiGraph(subgraph)
    
    def filter_by_category(self, category: str) -> 'WikiGraph':
        """
        Filter nodes by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            New WikiGraph with nodes in the specified category
        """
        def has_category(node_id: str, attrs: Dict[str, Any]) -> bool:
            categories = attrs.get('categories', [])
            return category in categories
        
        return self.filter_nodes(has_category)
    
    def filter_by_degree(self, min_degree: int = 0, max_degree: Optional[int] = None) -> 'WikiGraph':
        """
        Filter nodes by degree.
        
        Args:
            min_degree: Minimum total degree
            max_degree: Maximum total degree (optional)
            
        Returns:
            New WikiGraph with nodes in the specified degree range
        """
        def degree_in_range(node_id: str, attrs: Dict[str, Any]) -> bool:
            total_degree = self.graph.in_degree(node_id) + self.graph.out_degree(node_id)
            if max_degree is not None:
                return min_degree <= total_degree <= max_degree
            else:
                return min_degree <= total_degree
        
        return self.filter_nodes(degree_in_range)
    
    def get_largest_connected_component(self) -> 'WikiGraph':
        """
        Get the largest connected component of the graph.
        
        Returns:
            New WikiGraph with the largest connected component
        """
        # Get the undirected version of the graph to find connected components
        undirected = self.graph.to_undirected()
        
        # Get the largest connected component
        largest_cc = max(nx.connected_components(undirected), key=len)
        
        # Create subgraph with the nodes in the largest connected component
        subgraph = self.graph.subgraph(largest_cc).copy()
        
        return WikiGraph(subgraph)
    
    def get_layout(self, layout_type: str = 'spring', seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """
        Get node positions for visualization.
        
        Args:
            layout_type: Type of layout ('spring', 'circular', 'kamada_kawai', 
                                       'spectral', 'random', 'shell')
            seed: Random seed for reproducible layouts
            
        Returns:
            Dictionary mapping node ids to (x, y, z) positions
        """
        # Check if we have this layout cached
        cache_key = f"{layout_type}_{seed}"
        if cache_key in self._node_positions:
            return self._node_positions[cache_key]
        
        # Get the requested layout function
        if layout_type not in self._layouts:
            raise ValueError(f"Unknown layout type: {layout_type}")
        
        layout_func = self._layouts[layout_type]
        
        # Compute the layout
        pos = layout_func(seed=seed)
        
        # Cache the result
        self._node_positions[cache_key] = pos
        
        return pos
    
    def _spring_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute spring layout with 3D positions."""
        pos_2d = nx.spring_layout(self.graph, seed=seed)
        
        # Add a random z coordinate to make it 3D
        random.seed(seed)
        pos_3d = {
            node: (x, y, random.uniform(-0.5, 0.5)) 
            for node, (x, y) in pos_2d.items()
        }
        
        return self._normalize_positions(pos_3d)
    
    def _circular_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute circular layout with 3D positions."""
        pos_2d = nx.circular_layout(self.graph)
        
        # Place all nodes on the same z-plane initially
        pos_3d = {node: (x, y, 0.0) for node, (x, y) in pos_2d.items()}
        
        # Optionally, adjust z based on some node attribute
        # For example, use degree to determine height
        degrees = dict(self.graph.degree())
        min_degree = min(degrees.values()) if degrees else 0
        max_degree = max(degrees.values()) if degrees else 1
        
        # Avoid division by zero
        if max_degree > min_degree:
            for node in pos_3d:
                degree = degrees.get(node, min_degree)
                # Normalize to range [-0.5, 0.5]
                z = -0.5 + ((degree - min_degree) / (max_degree - min_degree))
                x, y, _ = pos_3d[node]
                pos_3d[node] = (x, y, z)
        
        return self._normalize_positions(pos_3d)
    
    def _kamada_kawai_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute Kamada-Kawai layout with 3D positions."""
        try:
            pos_2d = nx.kamada_kawai_layout(self.graph)
        except:
            # Fall back to spring layout if Kamada-Kawai fails
            return self._spring_layout(seed=seed)
        
        # Add z coordinate based on a metric like centrality
        centrality = nx.degree_centrality(self.graph)
        min_cent = min(centrality.values()) if centrality else 0
        max_cent = max(centrality.values()) if centrality else 1
        
        # Avoid division by zero
        if max_cent > min_cent:
            pos_3d = {
                node: (x, y, -0.5 + ((centrality.get(node, min_cent) - min_cent) / (max_cent - min_cent)))
                for node, (x, y) in pos_2d.items()
            }
        else:
            pos_3d = {node: (x, y, 0.0) for node, (x, y) in pos_2d.items()}
        
        return self._normalize_positions(pos_3d)
    
    def _spectral_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute spectral layout with 3D positions."""
        # For directed graphs, convert to undirected for spectral layout
        if isinstance(self.graph, nx.DiGraph):
            G = self.graph.to_undirected()
        else:
            G = self.graph
        
        try:
            # Try to compute a true 3D spectral layout if we have enough nodes
            if len(G) > 3:
                pos_3d = nx.spectral_layout(G, dim=3)
                return self._normalize_positions(pos_3d)
            else:
                # Fall back to 2D spectral with added z dimension
                pos_2d = nx.spectral_layout(G, dim=2)
                
                # Add z coordinate based on a random value
                random.seed(seed)
                pos_3d = {
                    node: (x, y, random.uniform(-0.5, 0.5)) 
                    for node, (x, y) in pos_2d.items()
                }
                
                return self._normalize_positions(pos_3d)
        except:
            # Fall back to spring layout if spectral layout fails
            return self._spring_layout(seed=seed)
    
    def _random_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute random layout with 3D positions."""
        random.seed(seed)
        pos_3d = {
            node: (
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0)
            ) 
            for node in self.graph.nodes()
        }
        
        return self._normalize_positions(pos_3d)
    
    def _shell_layout(self, seed: int = 42) -> Dict[str, Tuple[float, float, float]]:
        """Compute shell layout with 3D positions."""
        # Group nodes by some attribute (e.g., degree) to determine shells
        degrees = dict(self.graph.degree())
        if not degrees:
            return self._random_layout(seed=seed)
        
        # Define shells based on degree percentiles
        degree_values = sorted(degrees.values())
        n_shells = min(5, len(set(degree_values)))
        
        if n_shells <= 1:
            return self._circular_layout(seed=seed)
        
        percentiles = np.linspace(0, 100, n_shells + 1)[:-1]
        thresholds = [np.percentile(degree_values, p) for p in percentiles]
        
        # Create shells
        shells = [[] for _ in range(n_shells)]
        for node, degree in degrees.items():
            for i, threshold in enumerate(thresholds):
                if degree >= threshold:
                    shell_idx = i
            shells[shell_idx].append(node)
        
        # Remove empty shells
        shells = [shell for shell in shells if shell]
        
        if not shells:
            return self._circular_layout(seed=seed)
        
        # Compute 2D shell layout
        pos_2d = nx.shell_layout(self.graph, shells)
        
        # Add z coordinate based on shell index
        shell_dict = {}
        for i, shell in enumerate(shells):
            z_value = -0.5 + (i / max(1, len(shells) - 1))
            for node in shell:
                shell_dict[node] = z_value
        
        pos_3d = {
            node: (x, y, shell_dict.get(node, 0.0)) 
            for node, (x, y) in pos_2d.items()
        }
        
        return self._normalize_positions(pos_3d)
    
    def _normalize_positions(self, positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        """Normalize positions to be within the range [-1, 1] for all dimensions."""
        if not positions:
            return positions
        
        # Find min and max for each dimension
        x_vals = [x for x, _, _ in positions.values()]
        y_vals = [y for _, y, _ in positions.values()]
        z_vals = [z for _, _, z in positions.values()]
        
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        z_min, z_max = min(z_vals), max(z_vals)
        
        # Avoid division by zero for a dimension with all same values
        x_range = max(1e-6, x_max - x_min)
        y_range = max(1e-6, y_max - y_min)
        z_range = max(1e-6, z_max - z_min)
        
        # Normalize to [-1, 1]
        normalized = {}
        for node, (x, y, z) in positions.items():
            x_norm = -1 + 2 * ((x - x_min) / x_range)
            y_norm = -1 + 2 * ((y - y_min) / y_range)
            z_norm = -1 + 2 * ((z - z_min) / z_range)
            normalized[node] = (x_norm, y_norm, z_norm)
        
        return normalized
    
    def get_subgraph_between_nodes(self, 
                                  source: str, 
                                  target: str, 
                                  max_path_length: int = 3) -> Optional['WikiGraph']:
        """
        Get a subgraph between two nodes, including all paths up to max_path_length.
        
        Args:
            source: ID of the source node
            target: ID of the target node
            max_path_length: Maximum path length to include
            
        Returns:
            New WikiGraph with the subgraph, or None if no paths exist
        """
        # Check if nodes exist
        if source not in self.graph or target not in self.graph:
            return None
        
        # Get all simple paths up to max_path_length
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source=source, target=target, cutoff=max_path_length
            ))
        except nx.NetworkXNoPath:
            return None
        
        if not paths:
            return None
        
        # Get all nodes in the paths
        nodes = set()
        for path in paths:
            nodes.update(path)
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes).copy()
        
        return WikiGraph(subgraph)
    
    def get_neighbor_subgraph(self, 
                             center_node: str, 
                             radius: int = 1) -> Optional['WikiGraph']:
        """
        Get a subgraph centered on a node, including all neighbors up to specified radius.
        
        Args:
            center_node: ID of the center node
            radius: How many hops away to include
            
        Returns:
            New WikiGraph with the neighborhood subgraph, or None if node doesn't exist
        """
        if center_node not in self.graph:
            return None
        
        # Get ego network (a graph formed by a central node and its neighbors)
        nodes = {center_node}
        
        # Add nodes at distance 1, 2, ..., radius
        frontier = {center_node}
        for _ in range(radius):
            new_frontier = set()
            for node in frontier:
                # Add both incoming and outgoing neighbors
                in_neighbors = set(self.graph.predecessors(node))
                out_neighbors = set(self.graph.successors(node))
                new_frontier.update(in_neighbors.union(out_neighbors) - nodes)
            nodes.update(new_frontier)
            frontier = new_frontier
            if not frontier:
                break
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes).copy()
        
        return WikiGraph(subgraph)
