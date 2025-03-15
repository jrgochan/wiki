#!/bin/bash
# Setup script for creating a virtual environment for the Wikipedia 3D Graph Visualizer

# Set the name of the virtual environment
VENV_NAME=".venv"
REQUIREMENTS_FILE="requirements.txt"

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up virtual environment for Wikipedia 3D Graph Visualizer...${NC}"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Create the virtual environment
echo "Creating virtual environment at ${VENV_NAME}..."
python3 -m venv ${VENV_NAME}

# Check if venv creation was successful
if [ ! -d "${VENV_NAME}" ]; then
    echo "Failed to create virtual environment. Please check your Python installation."
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source ${VENV_NAME}/bin/activate

# Check if activation was successful
if [ -z "${VIRTUAL_ENV}" ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Install dependencies
echo "Installing dependencies from ${REQUIREMENTS_FILE}..."
pip install --upgrade pip
pip install -r ${REQUIREMENTS_FILE}

# Verify installation
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Virtual environment setup complete!${NC}"
    echo ""
    echo -e "${YELLOW}To activate the virtual environment, run:${NC}"
    echo "  source ${VENV_NAME}/bin/activate"
    echo ""
    echo -e "${YELLOW}To start the application, run:${NC}"
    echo "  python main.py"
    echo ""
    echo -e "${YELLOW}To generate a sample dataset, run:${NC}"
    echo "  python generate_sample.py"
    echo ""
    echo -e "${YELLOW}To deactivate the virtual environment when done, run:${NC}"
    echo "  deactivate"
else
    echo "Failed to install dependencies."
    exit 1
fi
