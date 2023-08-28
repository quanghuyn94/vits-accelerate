import requests
from tqdm import tqdm

import requests
from tqdm import tqdm
import os

def download_file(url, save_to_dir, filename=None):
    """
    Download a file from a specific URL and save it to the specified filename.

    Args:
        url (str): The URL path to the file to be downloaded.
        filename (str, option): The name of the file after downloading. If not provided, the filename will be automatically extracted from the URL.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    if not filename:
        filename = url.split("/")[-1]

    with open(os.path.join(save_to_dir, filename), "wb") as file, tqdm(
        desc=f'Download file: {filename}',
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=8192):
            bar.update(len(data))
            file.write(data)

    print(f"Save '{filename}' to cache")

def download_pretrain():
    D_0_URL = "https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth" 
    G_0_URL = "https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth" 
    os.makedirs("cache", exist_ok=True)
    download_file(D_0_URL, 'cache')
    download_file(G_0_URL, 'cache')
