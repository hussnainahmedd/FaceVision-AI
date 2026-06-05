"""
download_models.py — Download pre-trained age/gender estimation models.

Run this script once after cloning the repository:
    python download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

MODELS = [
    {
        "url": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/deploy.prototxt",
        "file": "age_deploy.prototxt",
    },
    {
        "url": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel",
        "file": "age_net.caffemodel",
    },
    {
        "url": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt",
        "file": "gender_deploy.prototxt",
    },
    {
        "url": "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel",
        "file": "gender_net.caffemodel",
    },
]


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["file"])
        if os.path.isfile(dest):
            print(f"  [OK] {model['file']} already exists, skipping.")
            continue

        print(f"  Downloading {model['file']} ...")
        try:
            urllib.request.urlretrieve(model["url"], dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"  [OK] {model['file']} ({size_mb:.1f} MB)")
        except Exception as exc:
            print(f"  [ERROR] Failed to download {model['file']}: {exc}")

    print("\nDone! You can now run: python app.py")


if __name__ == "__main__":
    print("Downloading age/gender estimation models...\n")
    main()
