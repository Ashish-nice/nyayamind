import logging
import os
import sys

def setup_logger(args):
    """
    Sets up the logging configuration for the training process.
    
    Args:
        args: ArgumentParser object containing 'logging_dir' and 'run_name'.
        
    Returns:
        logger: A configured logging object.
    """
    # Ensure the logging directory exists
    os.makedirs(args.logging_dir, exist_ok=True)

    # Define the log file path
    log_file_path = os.path.join(args.logging_dir, f"{args.run_name}_FT_logs.txt")

    file_handler = logging.FileHandler(log_file_path)
    stream_handler = logging.StreamHandler(sys.stdout)

    # Configure the logging format and handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            file_handler, stream_handler
        ]
    )

    logger = logging.getLogger(__name__)
    
    # Immediate feedback
    logger.info(f"Logging configured. Saving logs to: {log_file_path}")
    logger.info(f"Run Name: {args.run_name}")
    
    return logger