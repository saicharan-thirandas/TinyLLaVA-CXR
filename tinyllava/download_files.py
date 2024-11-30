from huggingface_hub import hf_hub_download
import os

# Repository details
repo_id = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"
repo_revision = "main"  # Use "main" or specific branch/revision
output_dir = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"  # Local directory to save files

# Function to download all files in the repository
def download_all_files(repo_id, repo_revision, output_dir):
    from huggingface_hub import list_repo_files

    # Get the list of all files in the repository
    files = list_repo_files(repo_id, revision=repo_revision)
    print(f"Found {len(files)} files to download...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download each file
    for file in files:
        print(f"Downloading {file}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            revision=repo_revision,
            cache_dir=output_dir
        )
    print(f"All files downloaded to {output_dir}")

# Run the script
download_all_files(repo_id, repo_revision, output_dir)
