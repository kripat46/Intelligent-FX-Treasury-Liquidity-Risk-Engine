# Kubernetes manifests

Deploys the API and dashboard as separate Deployments/Services, each
reading shared config from a ConfigMap.

## What's here

| File | Resource |
|---|---|
| `namespace.yaml` | `treasury-engine` namespace |
| `configmap.yaml` | shared env vars (`DATA_DIR`, `MLFLOW_TRACKING_URI`) |
| `api-deployment.yaml` | FastAPI serving layer, 1 replica |
| `api-service.yaml` | ClusterIP Service for the API |
| `api-hpa.yaml` | HorizontalPodAutoscaler — **pinned to 1 replica**, see below |
| `dashboard-deployment.yaml` | Streamlit dashboard, 1 replica |
| `dashboard-service.yaml` | LoadBalancer Service for the dashboard |

## Before you apply these

**Easiest path — Docker Desktop's built-in Kubernetes** (recommended if
you're already on Docker Desktop, which you are if you followed this
project's Docker Compose setup): Settings → **Kubernetes** → check
**Enable Kubernetes** → Apply & Restart. This gives you a real local
single-node cluster that shares Docker Desktop's image store directly —
your `treasury-api:latest`, `treasury-dashboard:latest`, and
`treasury-mlflow:latest` images (already built if you've run
`docker compose up --build`) are usable immediately, no registry push
needed. The manifests below already reference these local tags.

```bash
kubectl config get-contexts        # confirm "docker-desktop" is available
kubectl config use-context docker-desktop

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl get pods -n treasury-engine -w      # watch until both show Running
kubectl get svc -n treasury-engine
```

The dashboard's `Service` is `type: LoadBalancer` — on Docker Desktop's
local cluster this resolves to `localhost`, so once the pod is `Running`:

```
http://localhost:80          # or whatever port `kubectl get svc` shows
```

If a port conflict or `<pending>` external IP shows up, port-forward
directly instead, which always works regardless of Service type:

```bash
kubectl port-forward -n treasury-engine svc/treasury-dashboard 8501:80
kubectl port-forward -n treasury-engine svc/treasury-api 8000:80
```

**If instead you're deploying to a real remote cluster** (EKS/GKE/AKS),
you do need a registry:
```bash
docker build -f api/Dockerfile -t ghcr.io/<you>/treasury-api:latest .
docker build -f dashboard/Dockerfile -t ghcr.io/<you>/treasury-dashboard:latest .
docker push ghcr.io/<you>/treasury-api:latest
docker push ghcr.io/<you>/treasury-dashboard:latest
```
and change the `image:` field in both Deployment files from
`treasury-api:latest` / `treasury-dashboard:latest` to your pushed,
fully-qualified paths.

## Why the API is pinned to 1 replica (`minReplicas`/`maxReplicas` = 1)

The streaming feature store (`src/streaming_engine.py`) keeps its rolling
per-user / per-device history **in-process, in memory**. That's fine for a
single replica. The moment you scale the API Deployment past 1 replica
behind a plain round-robin Service, two consecutive requests for the same
user can land on different pods, each with a different (incomplete) view
of that user's recent history — the model would silently see a "colder"
user than it actually is, degrading feature quality without throwing any
error. That's a worse failure mode than just being slow, because nothing
tells you it happened.

The correct fix before scaling out is to externalize the feature store —
back `StreamingFeatureStore` with Redis (or a proper feature-store
service) keyed by `user_id`/`device_ip`, so every replica reads the same
rolling state. That's a legitimate next iteration on this project, not
implemented here — the honest move was to ship the HPA pinned at 1 rather
than a "looks production-ready" autoscaler that would misbehave the first
time it actually scaled.

The dashboard has no such constraint (each user's browser session owns
its own `st.session_state`), so it's a normal candidate for horizontal
scaling if you need it.

## What's verified vs. not

These manifests were written and validated for **YAML syntax correctness**
(`yaml.safe_load` on every file) and cross-checked for consistent naming
(Service selectors match Deployment labels, ConfigMap keys match what the
apps read via `os.environ`). They were **not** applied to a live cluster
from the environment this project was built in — there was no cluster
available to apply them to. Run `kubectl apply --dry-run=server -f k8s/`
against a real cluster (e.g. a local `kind` or `minikube`) as your first
real check before treating this as verified end-to-end.
