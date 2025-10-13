import os
import pandas as pd
import ml_monitoring_service.configuration as conf
from datetime import timedelta, datetime
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

def combine_services(task: str, active_set: str, age_latest_data: Optional[datetime]) -> None:
    """Combine Prometheus and Splunk data for the specified service set
    
    Args:
        task: Task type ('training' or 'inference')
        active_set: Name of the service set to combine data for
        age_latest_data: Timestamp of the latest existing data, or None if no existing data
    """

    active_services = conf.get_services(active_set)
    
    # Get JSON files only for active services and metrics
    json_files = []
    base_output_path = Path("output") / active_set / f"prometheus_data_{task}"
    
    for metric in conf.get_metrics(active_set):
        for service in active_services:
            service_clean = service.replace('-', '_')
            service_file_path = base_output_path / service_clean / f"{metric}-{service_clean}.json"
            
            if service_file_path.exists():
                json_files.append(str(service_file_path))
    
    if not json_files:
        logger.warning(f"No JSON files found for task '{task}' and active_set '{active_set}'")
        return

    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through each JSON file
    for json_file in json_files:
        try:
            # Read the JSON file into a DataFrame
            json_path = Path(json_file)
            service_name = json_path.stem.split('-')[1].replace('_', '-')
            
            df = pd.read_json(json_file)
            
            if df.empty:
                logger.warning(f"Empty dataframe from file: {json_file}")
                continue
            
            # Extract service name from the file name
            df['service'] = service_name
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            # Append the DataFrame to the list
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Error reading file {json_file}: {e}")
            continue

    # Concatenate all DataFrames along the rows
    if not dataframes:
        logger.error(f"No valid dataframes to concatenate for task '{task}' and active_set '{active_set}'")
        raise ValueError("No dataframes to concatenate")
    
    combined_prometheus_df = pd.concat(dataframes, axis=0, ignore_index=True, copy=False)

    # Group by 'timestamp' and 'service' and aggregate using 'first' to combine metrics into a single row
    combined_prometheus_df = combined_prometheus_df.groupby(['timestamp', 'service'], as_index=False).first()

    ############### COMBINE WITH SPLUNK DATA ########################################
    splunk_file = Path("output") / active_set / f"splunk_results_{task}.json"
    
    if not splunk_file.exists():
        logger.error(f"Splunk data file not found: {splunk_file}")
        raise FileNotFoundError(f"Splunk data file not found: {splunk_file}")
    
    try:
        splunk_data = pd.read_json(splunk_file, convert_dates=False)
    except Exception as e:
        logger.error(f"Error reading Splunk data from {splunk_file}: {e}")
        raise

    # Ensure 'timestamp' column in splunk_data is in the desired format
    splunk_data['timestamp'] = pd.to_datetime(splunk_data['timestamp'], errors='coerce')
    
    # Drop rows with invalid timestamps
    splunk_data = splunk_data.dropna(subset=['timestamp'])

    splunk_data['timestamp_nanoseconds'] = splunk_data['timestamp']

    # Round the timestamp in splunk_data to milliseconds
    splunk_data['timestamp'] = splunk_data['timestamp'].dt.round('ms')

    # Merge the DataFrames on the 'timestamp_ms' and 'service' columns
    new_dataset = pd.merge(combined_prometheus_df, splunk_data, on=['timestamp', 'service'], how='outer')

    output_file_path = Path("output") / active_set / f"{task}_dataset.json"
    
    # For inference_dataset, we don't concatenate with existing data, just use the new dataset
    if task == "inference":
        dataset = new_dataset
    # For other datasets (like training_dataset), check if the output file exists and concatenate
    elif output_file_path.exists():
        try:
            # Read existing data
            existing_dataset = pd.read_json(output_file_path)
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(existing_dataset['timestamp']):
                existing_dataset['timestamp'] = pd.to_datetime(existing_dataset['timestamp'], errors='coerce')
                existing_dataset = existing_dataset.dropna(subset=['timestamp'])
                
            # Concatenate existing data with new data
            dataset = pd.concat([existing_dataset, new_dataset], axis=0, ignore_index=True, copy=False)
            
            # Remove duplicates based on timestamp and service
            dataset = dataset.drop_duplicates(subset=['timestamp', 'service'], keep='last')
            
            # If this is the training dataset, filter out old data based on lookback hours
            if task == "training" and age_latest_data is not None:
                # Calculate the cutoff time based on training_lookback_hours
                current_time = datetime.now()
                cutoff_time = current_time - timedelta(hours=conf.get_training_lookback_hours(active_set))
                
                # Filter out data older than the cutoff time
                dataset = dataset[dataset['timestamp'] >= cutoff_time]
                logger.info(f"Filtered training data to {len(dataset)} rows (cutoff time: {cutoff_time})")
        except Exception as e:
            logger.error(f"Error processing existing dataset from {output_file_path}: {e}")
            logger.info("Using only new dataset")
            dataset = new_dataset
    else:
        dataset = new_dataset
    
    # Extract time features from timestamp
    dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
    dataset['hour'] = dataset['timestamp'].dt.hour
    dataset['minute'] = dataset['timestamp'].dt.minute
    
    logger.info(f"Combined dataset shape: {dataset.shape}")
    
    # Ensure output directory exists
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset.to_json(output_file_path, orient='records', date_format='iso', date_unit='us')
        logger.info(f"Successfully saved combined dataset to {output_file_path}")
    except Exception as e:
        logger.error(f"Error saving combined dataset to {output_file_path}: {e}")
        raise

if __name__ == "__main__":
    combine_services("main", "combined_dataset", None)
