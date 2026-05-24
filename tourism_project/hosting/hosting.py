# Pushes all deployment files to the Hugging Face Space (Streamlit frontend)
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# TODO: replace <your-hf-username> below
api.upload_folder(
    folder_path="tourism_project/deployment",   # local folder containing the app files
    repo_id="Kman42/wellness-tourism-prediction",  # the target Space
    repo_type="space",                           # dataset, model, or space
    path_in_repo="",                             # upload to the root of the Space
)
print("Deployment files pushed to the Hugging Face Space.")
