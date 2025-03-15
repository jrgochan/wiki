# Wikipedia 3D Graph Visualizer

An interactive application for visualizing Wikipedia articles as a 3D graph, where each node represents an article and each edge represents a hyperlink between articles.

## Project Structure

The main code is in the `wiki_graph` directory:

```
wiki_graph/
├── app.py                  # Dash web application
├── main.py                 # Command-line entry point
├── generate_sample.py      # Sample data generator
├── requirements.txt        # Project dependencies
├── setup_venv.sh           # Unix/macOS setup script
├── setup_venv.bat          # Windows setup script
├── README.md               # Project documentation
├── data/                   # Data storage directory
│   └── ...                 # JSON files for nodes and edges
└── modules/                # Core modules
    ├── wiki_api.py         # Wikipedia API integration
    ├── graph_utils.py      # Graph construction and analysis
    └── visualization.py    # 3D visualization components
```

## Features

- Fetch articles from Wikipedia using the Wikipedia API
- Build a graph from seed articles and their linked articles
- Generate synthetic datasets for demonstration
- Interactive 3D visualization using Plotly
- Multiple layout algorithms
- Customizable node and edge appearance
- Filtering by degree, category, or search term
- Find and highlight the shortest path between any two articles

## Setup and Installation

### Using Setup Scripts

**On Linux/macOS:**
```bash
cd wiki_graph
./setup_venv.sh
source .venv/bin/activate
```

**On Windows:**
```bash
cd wiki_graph
setup_venv.bat
```

### Manual Installation

```bash
cd wiki_graph
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Application

```bash
cd wiki_graph
python main.py
```

Then open your browser to http://localhost:8050 to access the web interface.

## Documentation

Detailed documentation can be found in the `wiki_graph/README.md` file.
