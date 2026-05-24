# Registers the raw dataset folder on the Hugging Face Hub (as a dataset repo)
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# TODO: replace <your-hf-username> with your Hugging Face username
repo_id = "Kman42/wellness-tourism-dataset"
repo_type = "dataset"

# Authenticate using the HF_TOKEN environment variable (set as a GitHub secret in CI)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check whether the dataset repo already exists; create it if not
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Step 2: Upload the local data folder to the dataset repo
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Dataset uploaded to the Hugging Face Hub.")
