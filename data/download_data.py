import io
import os
import zipfile
import requests

DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"


def download_data(target_dir="data"):

    data_path = os.path.join(target_dir, "PennFudanPed")
    if os.path.exists(data_path) and len(os.listdir(data_path)) > 0:
        print("[*] Dataset già presente, salto il download.")
        return data_path

    os.makedirs(target_dir, exist_ok=True)
    print("[*] Download in corso...")

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    response = requests.get(DATASET_URL, headers=headers)
    response.raise_for_status()

    print("[*] Decompressione archivio in memoria...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(target_dir)

    print("[*] File estratti con successo in:", data_path)
    return data_path


if __name__ == "__main__":
    download_data()