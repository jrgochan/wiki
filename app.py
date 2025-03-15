"""
Main application file for the Wikipedia 3D Graph Visualization tool.
"""

import os
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx

from modules.wiki_api import WikipediaDataFetcher
from modules.graph_utils import WikiGraph
from modules.visualization import GraphVisualizer

# Initialize the app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)
app.title = "Wikipedia 3D Graph Visualizer"
server = app.server

# Initialize global variables
wiki_fetcher = WikipediaDataFetcher()
wiki_graph = None
graph_viz = None

# Check if we have sample data
sample_data_exists = os.path.exists(os.path.join(wiki_fetcher.data_dir, "wiki_graph_synthetic_nodes.json"))

# Default list of seed articles
DEFAULT_SEEDS = [
    "Python (programming language)",
    "Machine learning",
    "Artificial intelligence",
    "Data science",
    "Computer science",
]

# Layout options
LAYOUT_OPTIONS = [
    {'label': 'Spring Layout', 'value': 'spring'},
    {'label': 'Circular Layout', 'value': 'circular'},
    {'label': 'Kamada-Kawai Layout', 'value': 'kamada_kawai'},
    {'label': 'Spectral Layout', 'value': 'spectral'},
    {'label': 'Random Layout', 'value': 'random'},
    {'label': 'Shell Layout', 'value': 'shell'},
]

# Create the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Wikipedia 3D Graph Visualizer", className="mt-4 mb-4"),
            html.P("Visualize and explore connections between Wikipedia articles in 3D space"),
        ], width=12)
    ]),
    
    # Data acquisition section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Acquisition"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H5("Seed Articles"),
                            dbc.Textarea(
                                id="seed-articles",
                                value="\n".join(DEFAULT_SEEDS),
                                placeholder="Enter article titles, one per line",
                                style={"height": "120px"},
                            ),
                            dbc.FormText("Specify the starting Wikipedia articles to build the graph from."),
                        ], width=6),
                        dbc.Col([
                            html.H5("Graph Parameters"),
                            dbc.InputGroup([
                                dbc.InputGroupText("Max Articles"),
                                dbc.Input(id="max-articles", type="number", value=50, min=5, max=300),
                            ], className="mb-2"),
                            dbc.InputGroup([
                                dbc.InputGroupText("Depth"),
                                dbc.Input(id="crawl-depth", type="number", value=2, min=1, max=3),
                            ], className="mb-2"),
                            dbc.FormText("Control the size and depth of the graph."),
                            dbc.Button(
                                "Fetch Wikipedia Data",
                                id="fetch-button",
                                color="primary",
                                className="mt-3",
                            ),
                            dbc.Button(
                                "Use Sample Dataset",
                                id="sample-button",
                                color="secondary",
                                className="mt-3 ms-2",
                                disabled=not sample_data_exists,
                            ),
                        ], width=6),
                    ]),
                ]),
            ], className="mb-3"),
        ], width=12),
    ]),
    
    # Status messages and graph info
    dbc.Row([
        dbc.Col([
            html.Div(id="status-message", className="alert alert-info"),
            html.Div(id="graph-info"),
        ], width=12),
    ]),
    
    # Visualization and controls
    dbc.Row([
        # Graph visualization column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("3D Graph Visualization"),
                dbc.CardBody([
                    dcc.Loading(
                        dcc.Graph(
                            id="graph-3d",
                            figure=go.Figure(),
                            style={"height": "70vh"},
                            config={
                                "displayModeBar": True,
                                "doubleClick": "autosize",
                                "displaylogo": False,
                            },
                        ),
                        type="cube",
                    ),
                ]),
            ]),
        ], width=9),
        
        # Controls column
        dbc.Col([
            # Visualization controls card
            dbc.Card([
                dbc.CardHeader("Visualization Controls"),
                dbc.CardBody([
                    html.H6("Layout Type"),
                    dcc.Dropdown(
                        id="layout-dropdown",
                        options=LAYOUT_OPTIONS,
                        value="spring",
                        clearable=False,
                    ),
                    
                    html.H6("Node Appearance", className="mt-3"),
                    dbc.InputGroup([
                        dbc.InputGroupText("Size"),
                        dbc.Input(id="node-size", type="number", value=10, min=1, max=50),
                    ], className="mb-2"),
                    
                    dbc.Checklist(
                        options=[
                            {"label": "Size nodes by degree", "value": "size_by_degree"},
                            {"label": "Color by centrality", "value": "color_by_centrality"},
                            {"label": "Show labels", "value": "show_labels"},
                        ],
                        value=[],
                        id="viz-options",
                        inline=False,
                        className="mb-2",
                    ),
                    
                    dbc.Button(
                        "Apply Visualization Settings",
                        id="apply-viz-button",
                        color="success",
                        className="mt-2",
                    ),
                ]),
            ], className="mb-3"),
            
            # Filtering card
            dbc.Card([
                dbc.CardHeader("Filtering"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.InputGroupText("Min Degree"),
                        dbc.Input(id="min-degree", type="number", value=0, min=0, max=100),
                    ], className="mb-2"),
                    
                    html.H6("Category Filter", className="mt-2"),
                    dcc.Dropdown(
                        id="category-dropdown",
                        options=[],
                        placeholder="Select a category",
                    ),
                    
                    dbc.Checklist(
                        options=[
                            {"label": "Show only largest connected component", "value": "connected_only"},
                        ],
                        value=[],
                        id="graph-filters",
                        inline=False,
                        className="mt-2",
                    ),
                    
                    dbc.Button(
                        "Apply Filters",
                        id="apply-filters-button",
                        color="info",
                        className="mt-2",
                    ),
                ]),
            ], className="mb-3"),
            
            # Degrees of separation card
            dbc.Card([
                dbc.CardHeader("Degrees of Separation"),
                dbc.CardBody([
                    dbc.InputGroup([
                        dbc.InputGroupText("From"),
                        dcc.Dropdown(
                            id="path-source-dropdown",
                            options=[],
                            placeholder="Select start article",
                            className="flex-grow-1",
                        ),
                    ], className="mb-2"),
                    
                    dbc.InputGroup([
                        dbc.InputGroupText("To"),
                        dcc.Dropdown(
                            id="path-target-dropdown",
                            options=[],
                            placeholder="Select end article",
                            className="flex-grow-1",
                        ),
                    ], className="mb-2"),
                    
                    dbc.Button(
                        "Find Path",
                        id="find-path-button",
                        color="warning",
                        className="mt-2",
                    ),
                    
                    html.Div(id="path-info", className="mt-2"),
                ]),
            ]),
        ], width=3),
    ]),
    
    # Hidden store components to share data between callbacks
    dcc.Store(id="graph-data-store"),
    dcc.Store(id="categories-store"),
    dcc.Interval(
        id="initial-load-interval",
        interval=500,  # in milliseconds
        n_intervals=0,
        max_intervals=1,  # Run only once
    ),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Wikipedia 3D Graph Visualizer - A tool for exploring connections between Wikipedia articles",
                className="text-muted text-center",
            ),
        ], width=12),
    ]),
], fluid=True)

# Callbacks

@app.callback(
    [Output("status-message", "children"),
     Output("status-message", "className"),
     Output("graph-data-store", "data"),
     Output("categories-store", "data")],
    [Input("fetch-button", "n_clicks"),
     Input("sample-button", "n_clicks")],
    [State("seed-articles", "value"),
     State("max-articles", "value"),
     State("crawl-depth", "value")],
    prevent_initial_call=True
)
def fetch_graph_data(fetch_clicks, sample_clicks, seed_articles_text, max_articles, depth):
    """Fetch graph data from Wikipedia or load sample data."""
    global wiki_graph, graph_viz
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return "No data loaded yet.", "alert alert-info", None, None
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        if button_id == "fetch-button":
            # Extract seed articles from text area
            seed_articles = [line.strip() for line in seed_articles_text.split("\n") if line.strip()]
            
            if not seed_articles:
                return "Please enter at least one seed article.", "alert alert-danger", None, None
            
            # Fetch data from Wikipedia
            message = f"Fetching data for {len(seed_articles)} seed articles (max: {max_articles}, depth: {depth})..."
            
            # Create the graph
            graph = wiki_fetcher.build_graph_from_seed(
                seed_articles=seed_articles,
                max_articles=max_articles,
                depth=depth,
                save_to_file=True
            )
            
            wiki_graph = WikiGraph(graph)
            graph_viz = GraphVisualizer(wiki_graph)
            
            message = f"Successfully built graph with {wiki_graph.get_node_count()} nodes and {wiki_graph.get_edge_count()} edges."
            alert_class = "alert alert-success"
            
        elif button_id == "sample-button":
            # Use sample dataset
            try:
                # First try to load an existing sample
                graph = wiki_fetcher.load_graph_from_file("wiki_graph_synthetic")
                message = "Loaded sample dataset from file."
            except FileNotFoundError:
                # If not found, generate a new sample
                graph = wiki_fetcher.generate_sample_dataset(size=100)
                message = "Generated new sample dataset."
            
            wiki_graph = WikiGraph(graph)
            graph_viz = GraphVisualizer(wiki_graph)
            
            message += f" Graph has {wiki_graph.get_node_count()} nodes and {wiki_graph.get_edge_count()} edges."
            alert_class = "alert alert-success"
        
        # Get categories for the dropdown
        categories = graph_viz.get_categories() if graph_viz else []
        
        return message, alert_class, "graph_loaded", categories
    
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return error_message, "alert alert-danger", None, None

@app.callback(
    [Output("graph-3d", "figure"),
     Output("graph-info", "children"),
     Output("path-source-dropdown", "options"),
     Output("path-target-dropdown", "options"),
     Output("category-dropdown", "options")],
    [Input("graph-data-store", "data"),
     Input("apply-viz-button", "n_clicks"),
     Input("apply-filters-button", "n_clicks"),
     Input("find-path-button", "n_clicks"),
     Input("categories-store", "data")],
    [State("layout-dropdown", "value"),
     State("node-size", "value"),
     State("viz-options", "value"),
     State("min-degree", "value"),
     State("category-dropdown", "value"),
     State("graph-filters", "value"),
     State("path-source-dropdown", "value"),
     State("path-target-dropdown", "value")],
    prevent_initial_call=True
)
def update_graph_visualization(graph_data, viz_clicks, filter_clicks, path_clicks, 
                              categories, layout, node_size, viz_options, 
                              min_degree, category, graph_filters, 
                              path_source, path_target):
    """Update the 3D graph visualization based on user settings."""
    global wiki_graph, graph_viz
    
    if graph_data is None or wiki_graph is None or graph_viz is None:
        # No data loaded yet
        return go.Figure(), "No graph data loaded.", [], [], []
    
    ctx = dash.callback_context
    if not ctx.triggered:
        # Initial load
        trigger_id = "graph-data-store"
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Update visualization settings
    if trigger_id in ["graph-data-store", "apply-viz-button", "categories-store"]:
        # Set layout
        graph_viz.set_layout(layout)
        
        # Update visual settings
        graph_viz.update_visual_settings(
            node_size=node_size,
            node_size_by_degree="size_by_degree" in viz_options,
            node_color_by_centrality="color_by_centrality" in viz_options,
            show_labels="show_labels" in viz_options,
        )
    
    # Update filters
    if trigger_id in ["graph-data-store", "apply-filters-button", "categories-store"]:
        graph_viz.update_filters(
            min_degree=min_degree,
            category=category,
            connected_component_only="connected_only" in graph_filters,
        )
    
    # Highlight path
    if trigger_id == "find-path-button" and path_source and path_target:
        path_found = graph_viz.highlight_path(path_source, path_target)
    
    # Generate the figure
    figure = graph_viz.get_figure()
    
    # Get graph statistics
    stats = graph_viz.get_graph_stats()
    
    # Prepare node dropdown options
    node_options = []
    for node, attrs in wiki_graph.get_all_node_attributes().items():
        title = attrs.get('title', node)
        node_options.append({
            'label': title,
            'value': node
        })
    
    # Sort options by label for better usability
    node_options.sort(key=lambda x: x['label'])
    
    # Prepare category dropdown options
    category_options = []
    if categories:
        category_options = [{'label': cat, 'value': cat} for cat in categories]
        category_options.sort(key=lambda x: x['label'])
    
    # Prepare graph info display
    info_elements = []
    
    # Basic graph stats
    info_elements.append(html.Div([
        html.Strong("Current Graph: "),
        html.Span(f"{stats['node_count']} nodes, {stats['edge_count']} edges"),
        html.Span(f" (filtered from original {stats['original_node_count']} nodes, {stats['original_edge_count']} edges)"),
    ]))
    
    # Path information if available
    if 'path_source' in stats:
        path_length = stats['path_length']
        degrees_text = f"{path_length} degree{'s' if path_length != 1 else ''} of separation"
        info_elements.append(html.Div([
            html.Strong("Path: "),
            html.Span(f"{stats['path_source']} → {stats['path_target']}"),
            html.Span(f" ({degrees_text})"),
        ]))
    
    graph_info = html.Div(info_elements, className="mb-3")
    
    return figure, graph_info, node_options, node_options, category_options

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
