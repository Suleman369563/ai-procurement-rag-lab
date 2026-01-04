# n8n Orchestration (Project 2)

This workflow runs the Python pipeline on the host from n8n in Docker via SSH.

## 1) Start n8n (Docker)
```bash
REPO_ROOT="/path/to/project-2-ai-procurement-agent"

docker run --rm -it --name n8n -p 5678:5678 \
  --add-host=host.docker.internal:host-gateway \
  -v "$REPO_ROOT/.n8n:/home/node/.n8n" \
  n8nio/n8n
```

If `host.docker.internal` is not available, use the Docker bridge IP (often `172.17.0.1`) in the SSH credential.

## 2) Create a dedicated SSH key
```bash
ssh-keygen -t ed25519 -f ~/.ssh/n8n_ed25519 -C "n8n" -N ""
cat ~/.ssh/n8n_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

## 3) Add SSH credentials in n8n
1. Open `http://localhost:5678`.
2. Create new credentials: **SSH**.
3. Host: `host.docker.internal` (or `172.17.0.1`).
4. Port: `22`.
5. User: your Linux username (for this repo, `ghost`).
6. Authentication: **Private Key**.
7. Paste the contents of `~/.ssh/n8n_ed25519` as the private key.

## 4) Import the workflow
- Import `n8n/procurement_orchestration.json`.
- Assign the SSH credential to each SSH node.
- The workflow uses `python -m rag.index` and `python -m rag.query` to ensure the package resolves correctly.

## 5) Run the workflow
- Ensure `.env` has `GOOGLE_API_KEY`.
- Execute the workflow from the manual trigger.

Notes:
- Update the SSH node commands if your repo path differs.
- n8n is orchestrating only; all business logic stays in the Python scripts.
