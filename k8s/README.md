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

1. Build and push the images to a registry these manifests can pull from:
   ```bash
   docker build -f api/Dockerfile -t ghcr.io/<you>/treasury-api:latest .
   docker build -f dashboard/Dockerfile -t ghcr.io/<you>/treasury-dashboard:latest .
   docker push ghcr.io/<you>/treasury-api:latest
   docker push ghcr.io/<you>/treasury-dashboard:latest
   ```
2. Replace `your-registry/treasury-api:latest` and
   `your-registry/treasury-dashboard:latest` in the two Deployment files
   with your actual pushed image paths.
3. Apply:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/
   kubectl get pods -n treasury-engine
   kubectl get svc treasury-dashboard -n treasury-engine   # find the external IP
   ```

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
