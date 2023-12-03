import os
import gdown


def download_model(output: str):
    os.makedirs("pruned_models", exist_ok=True)
    url = "https://drive.google.com/u/0/uc?id=1S89wGRcyIKlZqCuySu43Wfx3VKhkYCZn&export=download"
    gdown.download(url=url, output=output, quiet=False)

if __name__ == "__main__":
    download_model("./pruned_models/model_concatenation_large_params_pruned.h5")