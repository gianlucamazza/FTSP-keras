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


def check_existing_instances(api_key):
    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--api-key", api_key, "--raw"],
            capture_output=True,
            text=True,
            check=True
        )
        instances = json.loads(result.stdout)
        available_instances = [instance for instance in instances if instance.get('state') == 'running']
        if available_instances:
            logger.info(f"Using existing instance: {available_instances[0]['id']}")
            return available_instances[0]['id']
        else:
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during check_existing_instances: {e}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from Vast.ai output.")
        return None


def search_offers():
    try:
        query = "num_gpus>=2 reliability>0.99 rentable=True verified=True"
        result = subprocess.run(
            ["vastai", "search", "offers", query, "--raw"],
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
            ["vastai", "create", "instance", str(offer_id), "--api-key", api_key,
             "--image", image, "--disk", str(disk_size)],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Create instance command executed successfully")
        instance_details = json.loads(result.stdout)
        logger.info(f"Created instance: {instance_details}")
        return instance_details['id']
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during create_instance: {e}")
        logger.error(f"stderr: {e.stderr}")
        return None
    except json.JSONDecodeError:
        logger.error("Failed to decode JSON from Vast.ai output.")
        return None


def main():
    # Install Vast.ai client if not already installed
    install_vastai()

    # Read the Vast.ai API token from the file
    api_key_path = os.path.expanduser("~/.vast_api_key")
    api_key = read_api_key(api_key_path)

    # Check for existing instances
    instance_id = check_existing_instances(api_key)
    if instance_id:
        return instance_id

    # Search for offers with the specified criteria and select the most profitable one
    offer_id = search_offers()
    if not offer_id:
        logger.error("No suitable offer found.")
        return None

    # Define the image and disk size for the instance
    image = "tensorflow/tensorflow:latest-gpu"
    disk_size = 32

    # Create an instance with the selected offer ID
    return create_instance(api_key, offer_id, image, disk_size)


if __name__ == "__main__":
    instance_id = main()
    if instance_id:
        print(f"Instance ID: {instance_id}")
    else:
        print("Failed to start an instance.")
