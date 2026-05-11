import os
import subprocess
import urllib.request
import json
import sys

token = os.environ.get("GITHUB_TOKEN", "")

print("Authenticating with GitHub...")
req = urllib.request.Request("https://api.github.com/user")
req.add_header("Authorization", f"token {token}")
req.add_header("Accept", "application/vnd.github.v3+json")

try:
    with urllib.request.urlopen(req) as response:
        user_data = json.loads(response.read().decode())
        username = user_data["login"]
        print(f"Logged in as {username}")
except Exception as e:
    print(f"Authentication failed (Make sure your token is valid and has 'repo' scope): {e}")
    sys.exit(1)

print("\nCreating Indic-LLM-Benchmark repository on GitHub...")
req = urllib.request.Request("https://api.github.com/user/repos", data=json.dumps({
    "name": "Indic-LLM-Benchmark", 
    "description": "A multi-agent LangGraph pipeline for generating and evaluating benchmarks for Indian language LLMs.",
    "private": False
}).encode())
req.add_header("Authorization", f"token {token}")
req.add_header("Accept", "application/vnd.github.v3+json")

clone_url = ""
try:
    with urllib.request.urlopen(req) as response:
        repo_data = json.loads(response.read().decode())
        clone_url = repo_data["clone_url"]
        print(f"Repository successfully created: {clone_url}")
except urllib.error.HTTPError as e:
    if e.code == 422:
        print("Repository 'Indic-LLM-Benchmark' already exists on your account! Proceeding to push to it...")
        clone_url = f"https://github.com/{username}/Indic-LLM-Benchmark.git"
    else:
        print(f"Failed to create repo: {e}")
        sys.exit(1)

# Add token to URL for authentication
auth_url = clone_url.replace("https://", f"https://{token}@")

print("\nSetting up Git remote and pushing code...")
subprocess.run(["git", "remote", "remove", "origin"], stderr=subprocess.DEVNULL)
subprocess.run(["git", "remote", "add", "origin", auth_url], check=True)
subprocess.run(["git", "branch", "-M", "main"], check=True)

# Push to origin main
push_result = subprocess.run(["git", "push", "-u", "origin", "main"])

if push_result.returncode == 0:
    print("\n✅ Successfully pushed all code to GitHub!")
    print(f"Check it out here: https://github.com/{username}/Indic-LLM-Benchmark")
else:
    print("\n❌ Failed to push. Your token might not have the correct permissions.")

# Clean up remote URL so token is not saved in local git config
subprocess.run(["git", "remote", "remove", "origin"])
subprocess.run(["git", "remote", "add", "origin", clone_url])
