#!/usr/bin/python3

from huggingface_hub import login, snapshot_download
login(token = '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')
snapshot_download(repo_id = "meta-llama/Llama-2-7b-hf")

