import os
from huggingface_hub import login


def login_huggingface():
    """
    Logs in to Huggingface using the provided API token.
    Requires the HUGGINGFACEHUB_API_TOKEN environment variable to be set.
    """
    login(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    print("[Info] ~ Logged in to Huggingface.")
