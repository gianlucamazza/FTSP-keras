import os
import subprocess
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_vastai():
    subprocess.run(["pip", "install", "vastai", "--upgrade"], check=True)

def read_api_key(filepath):
    try:
        with open(filepath, 'r') as file:
            token = file.read().strip()
            if not token:
                raise ValueError("API key file is empty")
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at {filepath}")

def search_offers():
    try:
        result = subprocess.run(
            ["vastai", "search", "offers", "num_gpus >= 2", "-o", "num_gpus-", "--raw"],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Search offers command executed successfully")
        offers = json.loads(result.stdout)
        if not offers:
            raise ValueError("No offers found.")
        # Sort by profitability or other criteria
        offers.sort(key=lambda x: x['dlperf_per_dphtotal'], reverse=True)
        logger.info(f"Top offer: {offers[0]}")
        return offers[0]['id']
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during search_offers: {e}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from Vast.ai output.")
        return None
    except ValueError as e:
        logger.error(e)
        return None

def create_instance(api_key, offer_id, image, disk_size):
    try:
        result = subprocess.run(
            ["vastai", "create", "instance", str(offer_id), "--api-key", api_key, "--image", image, "--disk", str(disk_size)],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Create instance command executed successfully")
        logger.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during create_instance: {e}")
        logger.error(f"stderr: {e.stderr}")

def main():
    # Install Vast.ai client if not already installed
    install_vastai()

    # Read the Vast.ai API token from the file
    api_key_path = os.path.expanduser("~/.vast_api_key")
    api_key = read_api_key(api_key_path)

    # Search for offers with the specified criteria and select the most profitable one
    offer_id = search_offers()
    if not offer_id:
        logger.error("No suitable offer found.")
        return

    # Define the image and disk size for the instance
    image = "tensorflow/tensorflow:latest-gpu"
    disk_size = 32

    # Create an instance with the selected offer ID
    create_instance(api_key, offer_id, image, disk_size)

if __name__ == "__main__":
    main()
