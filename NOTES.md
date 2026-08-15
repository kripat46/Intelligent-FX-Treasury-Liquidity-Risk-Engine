# NOTES.md — Rerunning Docker & Kubernetes

Practical runbook only — not documentation for a reader, just copy-paste
commands for future-you (or for demoing live in an interview). All
commands are PowerShell, run from the project root:
`E:\Intelligent FX Engine Update (Docker)\wise_liquidity_engine`

---

## Part 1 — Docker Compose (the easy path)

This is the normal day-to-day way to run the project. No known issues —
this part just works.

```powershell
docker compose up -d          # detached, gives your terminal back immediately
docker compose ps             # confirm all 3 show "healthy" / "Up"
```

Access:
- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- MLflow UI: http://localhost:5000

Stop when done:
```powershell
docker compose down            # stops + removes containers, KEEPS volumes (data/models survive)
```

If you ever want a truly clean slate (re-triggers the ~10s bootstrap):
```powershell
docker compose down -v         # -v also wipes the named volumes
docker compose up -d --build
```

**If a container won't go healthy after a long time asleep:** rebuild
fresh rather than debugging a stale image —
```powershell
docker compose down
docker compose up -d --build
```

---

## Part 2 — Kubernetes (the involved path)

Budget ~15 minutes. This is the sequence that actually worked, including
the three real gotchas hit the first time — skip the trial-and-error and
just run these in order.

### Step 1 — Make sure Docker Desktop's Kubernetes is on

Docker Desktop → Settings → **Kubernetes** → Enable Kubernetes (kind
provisioning, default settings) → Apply & Restart. Takes a few minutes
first time; faster on subsequent enables.

Verify:
```powershell
kubectl config current-context     # should say: docker-desktop
```

### Step 2 — Rebuild the images if they don't already exist

```powershell
docker images | Select-String "treasury"
```
If `treasury-api:latest` and `treasury-dashboard:latest` aren't listed:
```powershell
docker compose build api dashboard
```

### Step 3 — Load images into the cluster (the actual fix that was needed)

**Why this step exists:** Docker Desktop's `kind`-based Kubernetes does
NOT automatically share Docker's image store, and the standalone `kind`
CLI (if installed) doesn't see Docker Desktop's cluster either — they're
separate implementations. Skipping this step is what caused
`ImagePullBackOff` the first time around.

```powershell
docker save treasury-api:latest -o treasury-api.tar
docker save treasury-dashboard:latest -o treasury-dashboard.tar

docker cp treasury-api.tar desktop-control-plane:/treasury-api.tar
docker cp treasury-dashboard.tar desktop-control-plane:/treasury-dashboard.tar

# NOTE: space between -n and k8s.io, NOT -n=k8s.io — the = form fails
# silently with a cryptic "No help topic for '.io'" error.
docker exec desktop-control-plane ctr -n k8s.io images import /treasury-api.tar
docker exec desktop-control-plane ctr -n k8s.io images import /treasury-dashboard.tar
```

Verify both actually landed before moving on:
```powershell
docker exec desktop-control-plane ctr -n k8s.io images ls | findstr treasury
```
You must see BOTH `treasury-api:latest` and `treasury-dashboard:latest`
in the output before proceeding. If you don't see both, stop and re-run
whichever `docker cp` / `ctr import` line is missing — do not apply the
manifests yet, it'll just cycle through `ImagePullBackOff` for nothing.

### Step 4 — Apply the manifests

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl get pods -n treasury-engine -w
```

Wait for both `treasury-api-...` and `treasury-dashboard-...` pods to
reach `1/1 Running`. Ctrl+C once they do. If they don't within ~60s and
show `ImagePullBackOff` instead, the manifests' `image:` field has
probably reverted to the placeholder — check:
```powershell
Select-String -Path k8s\api-deployment.yaml -Pattern "image:"
Select-String -Path k8s\dashboard-deployment.yaml -Pattern "image:"
```
Both must say `treasury-api:latest` / `treasury-dashboard:latest` — NOT
`your-registry/treasury-api:latest`. If wrong, fix the line and re-run
`kubectl apply -f k8s/` then `kubectl rollout restart deployment -n treasury-engine`.

### Step 5 — Reach the dashboard

```powershell
kubectl port-forward -n treasury-engine svc/treasury-dashboard 8501:80
```
Open http://localhost:8501

(Leave that port-forward command running in its own terminal — it's
blocking by design, streaming the connection.)

### Cleanup when done demoing

```powershell
kubectl delete namespace treasury-engine    # tears down everything cleanly
```
Kubernetes toggle can stay on or be turned back off in Docker Desktop
settings — doesn't matter either way, next `Enable Kubernetes` just
recreates the cluster from scratch regardless.

---

## Quick reference — known gotchas, in one place

| Symptom | Cause | Fix |
|---|---|---|
| `context deadline exceeded` on `docker compose up --build` | Docker Desktop's Bake builder flaking | `$env:COMPOSE_BAKE="false"` then retry, or build each image individually first |
| `mlflow-1` crash-loops a few times before stabilizing | Multiple MLflow workers racing to init the same SQLite schema | Already fixed in `docker-compose.yml` via `--workers 1` on the mlflow command — if you ever revert that, this comes back |
| `api` fails to start because `mlflow` is unhealthy | Old compose file had `api` hard-depend on `mlflow`, which `api` doesn't actually need | Already removed in current `docker-compose.yml` — don't re-add it |
| `kind get clusters` returns nothing / `kind load` says "no nodes found" | Standalone `kind` CLI is a different tool from Docker Desktop's internal kind — they don't share state | Use the `docker save` / `docker cp` / `ctr import` sequence in Step 3 instead, don't bother with standalone `kind` CLI at all |
| `ImagePullBackOff`, `kubectl describe pod` shows `your-registry/...` | Placeholder image name in the manifest, not the actual local image | Fix the `image:` line to `treasury-api:latest` / `treasury-dashboard:latest` |
| `ctr -n=k8s.io ...` → `No help topic for '.io'` | Wrong flag syntax for this `ctr` version | Use `-n k8s.io` (space, not `=`) |
| Streamlit dashboard shows `Failed to fetch dynamically imported module` after a rebuild | Stale browser cache referencing old JS bundle hashes | Hard refresh (`Ctrl+Shift+R`) or open in an incognito window |
