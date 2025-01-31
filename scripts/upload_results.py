from huggingface_hub import upload_file, list_repo_files
import os


def upload_folder_with_check(local_folder, repo_id, path_in_repo="", repo_type="dataset"):
    """
    Uploads an entire folder to a Hugging Face repository, skipping files that already exist.

    Args:
        local_folder (str): Path to the local folder to upload.
        repo_id (str): The Hugging Face repository ID (e.g., "username/repo_name").
        path_in_repo (str): Path in the repository to upload to (default is root).
        repo_type (str): Type of repository ("model", "dataset", or "space").
    """
    # Get a list of existing files in the repository
    print("Fetching existing files from the repository...")
    existing_files = list_repo_files(repo_id, repo_type=repo_type)
    print(f"Found {len(existing_files)} existing files in the repository.")

    for root, _, files in os.walk(local_folder):
        for file_name in files:
            local_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_path, local_folder)  # Path relative to the folder being uploaded
            repo_path = os.path.join(path_in_repo, relative_path).replace("\\", "/")  # Normalize for Hugging Face repo
            
            if repo_path in existing_files:
                print(f"Skipping {local_path}: File already exists in the repository as {repo_path}.")
                continue  # Skip if the file already exists
            
            print(f"Uploading {local_path} to {repo_path}...")
            upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type=repo_type,
            )


# Example usage
# local_folder = "/home/xpeng3/FinBen/results/nvidia__Llama-3.1-Minitron-4B-Depth-Base"
# repo_id = "TheFinAI/lm-eval-results-zh-5shot-private"  # Replace with your repo details
# path_in_repo = "nvidia__Llama-3.1-Minitron-4B-Depth-Base"
# upload_folder_with_check(local_folder, repo_id, path_in_repo)





upload_file(
    path_or_fileobj="/home/xpeng3/FinBen/results/nvidia__Llama-3.1-Minitron-4B-Width-Base/results_2025-01-31T14-34-16.023078.json",  # Local file path
    path_in_repo="nvidia/Llama-3.1-Minitron-4B-Width-Base/results_2025-01-31T14-34-16.023078.json",            # Desired file path in the repo
    repo_id="TheFinAI/lm-eval-results-zh-5shot-private",        # Replace with your repo details
    repo_type="dataset",                       # Use "model", "dataset", or "space" as appropriate
)

