import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("video_gen_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(name: str = __name__):
    """Get a logger with the specified name"""
    return logging.getLogger(name)