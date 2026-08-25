#!/usr/bin/env python3
"""
Automated Upstream PR & Issue Submission Tool for Turing Engine.
Forks upstream repos, prepares branches with Turing adapters, and submits PRs.
"""

import os
import sys
import subprocess
import argparse
import tempfile
import shutil

UPSTREAM_TARGETS = {
    "litellm": {
        "repo": "BerriAI/litellm",
        "branch": "feat/turing-engine-provider",
        "title": "feat(providers): Add native support for Turing Engine LLM serving runtime",
        "body_file": "integrations/litellm/UPSTREAM_PR.md",
        "source_file": "integrations/litellm/turing_engine.py",
        "dest_path": "litellm/llms/turing_engine.py"
    },
    "langchain": {
        "repo": "langchain-ai/langchain",
        "branch": "feat/turing-engine-chat-model",
        "title": "feat(community): Add ChatTuringEngine integration for Turing Engine",
        "body_file": "integrations/langchain/UPSTREAM_PR.md",
        "source_file": "integrations/langchain/chat_turing.py",
        "dest_path": "libs/community/langchain_community/chat_models/turing.py"
    },
    "llamaindex": {
        "repo": "run-llama/llama_index",
        "branch": "feat/turing-engine-llm-adapter",
        "title": "feat(llms): Add native Turing Engine serving adapter",
        "body_file": "integrations/llamaindex/UPSTREAM_PR.md",
        "source_file": "integrations/llamaindex/turing_llm.py",
        "dest_path": "llama-index-integrations/llms/llama-index-llms-turing/llama_index/llms/turing/base.py"
    },
    "kserve": {
        "repo": "kserve/kserve",
        "branch": "feat/turing-serving-runtime",
        "title": "feat(servingruntimes): Add Turing Engine high-throughput ServingRuntime",
        "body_file": "integrations/kserve/UPSTREAM_PR.md",
        "source_file": "integrations/kserve/serving_runtime.yaml",
        "dest_path": "config/runtimes/turing-runtime.yaml"
    }
}

def submit_pr(target_key: str, config: dict, dry_run: bool = False):
    repo = config["repo"]
    branch = config["branch"]
    title = config["title"]
    body_file = config["body_file"]
    source_file = config["source_file"]
    dest_path = config["dest_path"]

    print("=" * 80)
    print(f"   🚀 Upstream PR Submission: {repo}")
    print("=" * 80)

    if not os.path.exists(body_file) or not os.path.exists(source_file):
        print(f"❌ Error: Required integration files missing: {body_file} or {source_file}")
        return

    with open(body_file, "r") as f:
        body_content = f.read()

    print(f"Target Repo: {repo}")
    print(f"Branch:      {branch}")
    print(f"PR Title:    {title}")
    print(f"Source File: {source_file} -> {dest_path}\n")

    if dry_run:
        print("🔍 [DRY-RUN] Upstream PR validated. Would run:")
        print(f"  1. gh repo fork {repo} --clone=false")
        print(f"  2. git checkout -b {branch}")
        print(f"  3. copy {source_file} -> {dest_path}")
        print(f"  4. git commit -m '{title}'")
        print(f"  5. git push origin {branch}")
        print(f"  6. gh pr create --repo {repo} --title '{title}' --body-file {body_file}\n")
        return

    # Check gh auth
    try:
        gh_user = subprocess.check_output(["gh", "api", "user", "--jq", ".login"], text=True).strip()
    except Exception as e:
        print(f"❌ Failed to get authenticated GitHub user: {e}")
        return

    temp_dir = tempfile.mkdtemp(prefix=f"turing_upstream_{target_key}_")
    try:
        print(f"Forking and cloning {repo} into temporary workspace...")
        subprocess.run(["gh", "repo", "fork", repo, "--clone=true", temp_dir], check=True)

        subprocess.run(["git", "checkout", "-b", branch], cwd=temp_dir, check=True)

        target_file_full = os.path.join(temp_dir, dest_path)
        os.makedirs(os.path.dirname(target_file_full), exist_ok=True)
        shutil.copyfile(source_file, target_file_full)

        subprocess.run(["git", "add", dest_path], cwd=temp_dir, check=True)
        subprocess.run(["git", "commit", "-m", title], cwd=temp_dir, check=True)
        subprocess.run(["git", "push", "-u", "origin", branch], cwd=temp_dir, check=True)

        print(f"Creating pull request on {repo}...")
        pr_cmd = [
            "gh", "pr", "create",
            "--repo", repo,
            "--head", f"{gh_user}:{branch}",
            "--title", title,
            "--body", body_content
        ]
        res = subprocess.run(pr_cmd, cwd=temp_dir, capture_output=True, text=True, check=True)
        print(f"✅ Successfully created Upstream PR: {res.stdout.strip()}\n")

    except Exception as e:
        print(f"❌ Error submitting upstream PR to {repo}: {e}\n")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Submit upstream PRs for Turing Engine integrations.")
    parser.add_argument("--target", choices=["litellm", "langchain", "llamaindex", "kserve", "all"], default="all", help="Target upstream integration.")
    parser.add_argument("--dry-run", action="store_true", help="Preview PR submission steps without modifying upstream repos.")
    args = parser.parse_args()

    targets = UPSTREAM_TARGETS.keys() if args.target == "all" else [args.target]
    for t in targets:
        submit_pr(t, UPSTREAM_TARGETS[t], dry_run=args.dry_run)

if __name__ == "__main__":
    main()
