import logging
import networkx as nx
import matplotlib
import matplotlib.pyplot
import os
import mlflow
import pandas as pd
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

matplotlib.use('Agg')

def save_data_to_mlflow(df: pd.DataFrame, file_name: str, active_set: str) -> str:
    """
    Save the DataFrame to MLflow as an artifact.
    
    Args:
        df (pandas.DataFrame): The DataFrame to save
        file_name (str): Name to give the saved file
        active_set (str): The active service set name
    
    Returns:
        str: Path to the saved file
    """
    logger.info(f"Saving {file_name} to MLflow...")
    
    # Save DataFrame to a temporary JSON file
    temp_path = f"output/{active_set}/{file_name}"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    # Save to JSON with proper orientation for easier viewing
    df.to_json(temp_path, orient="records", date_format="iso")
    
    # Log the file as an artifact in MLflow
    if mlflow.active_run():
        mlflow.log_artifact(temp_path)
        logger.info(f"Successfully logged {file_name} to MLflow")
    else:
        logger.warning("No active MLflow run found. File saved locally but not logged to MLflow.")
    
    return temp_path

def visualize_microservice_graph(
    service_relationships: Dict[str, List[str]], 
    result_graph_path: str, 
    service_errors: Optional[Dict[str, float]] = None
) -> None:
    """Visualize the microservice graph using networkx and matplotlib with dark theme
    
    Args:
        service_relationships: Dictionary mapping services to their related services
        result_graph_path: Path where the visualization should be saved
        service_errors: Optional dictionary mapping service names to error values for coloring
    """
    # Close any existing plots and create new figure with black background
    matplotlib.pyplot.close('all')
    
    # Create figure with explicit black background
    fig = matplotlib.pyplot.figure(facecolor='black', figsize=(10, 7))
    ax = fig.add_subplot(111, facecolor='black')
    
    G = nx.DiGraph()
    
    # Add nodes and edges
    for service, related_services in service_relationships.items():
        G.add_node(service)
        for related_service in related_services:
            G.add_edge(service, related_service)
    
    # Position mobile-bff at the top
    pos = nx.spring_layout(G)
    if 'mobile-bff' in pos:
        pos['mobile-bff'][1] += 1.5
    
    # Node colors based on errors
    if service_errors and any(service_errors.values()):
        max_error = max(service_errors.values())
        min_error = min(service_errors.values())
        node_colors = [
            matplotlib.pyplot.cm.OrRd((service_errors[service] - min_error) / (max_error - min_error))
            if service in service_errors else '#4FB6D6'
            for service in G.nodes
        ]
    else:
        node_colors = '#4FB6D6'
    
    # Modify labels to split names and remove hyphens
    labels = {service: service.replace('-', ' ').replace(' ', '\n') for service in G.nodes}
    
    # Draw the graph
    nx.draw(
        G, 
        pos,
        ax=ax,
        labels=labels,
        with_labels=True,
        node_size=6000,
        node_color=node_colors,
        font_size=11,
        font_color='black',
        font_weight='bold',
        arrows=True,
        edge_color='#FFFFFF',
        arrowsize=30,
        # width=2,
        alpha=0.99
    )
    
    matplotlib.pyplot.title("Microservice Graph", color='white', pad=40)
    
    # Add explanatory text
    matplotlib.pyplot.text(0.5, 0.02, "Service(s) with high reconstruction error have stronger red color", 
             ha='center', va='center', transform=ax.transAxes, color='red', fontsize=12)
    
    # Remove axes and set background limits
    ax.set_axis_off()
    ax.set_facecolor('black')
    ax.set_alpha(1.0)
    
    # Set the figure background
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(1.0)
    
    # Adjust the plot to fill the figure
    matplotlib.pyplot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Display the plot
    matplotlib.pyplot.savefig(result_graph_path, facecolor=fig.get_facecolor(), edgecolor='none')
    matplotlib.pyplot.close(fig)
