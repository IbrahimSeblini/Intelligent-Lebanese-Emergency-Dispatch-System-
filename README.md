# Emergency dispatch classifier

This project is an **emergency intake and dispatch stack** aimed at Lebanon’s unified emergency context. Incoming reports (for example from a **voice / telephony webhook** or **WhatsApp**) are stored and enriched: a **classifier service** decides which services are needed (police, fire, medical), and the **dispatcher API** coordinates **severity assessment**, **caller-facing responses**, and **PostgreSQL** persistence. Operators use **web dashboards** (main dispatcher plus per-service views) to see and manage calls; **Grafana** and **Prometheus metrics** support monitoring.

The classifier API exposes `/api/dispatch` (hybrid NLP: rules + sklearn in typical deployment). The dispatcher exposes `/webhook` for intake, `/calls` for listing and updates, plus `/health` and `/metrics`.

---

## Using the deployed environment

All URLs below match the current **[`links.txt`](links.txt)** file in this repo **BUT DO NOT WORK ANYMORE, AS THIS WAS A UNIVERSITY TEMPORARY PROJECT, LINKS ARE KEPT AS THEY ARE FOR COMPLETION**. If infrastructure or tunnels are recreated, update that file and adjust these links accordingly.

**Base URL (load balancer):**  
`http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com`

### Dashboards (open in a browser)

| Purpose | URL |
|--------|-----|
| Main dispatcher | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/dispatch-dashboard.html |
| Fire | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/fire-dashboard.html |
| Police | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/police-dashboard.html |
| Medical (Red Cross) | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/red-cross-dashboard.html |
| Service overview | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/service-dashboard.html |

### APIs and health (no local install required)

| What | URL |
|------|-----|
| List calls (JSON for dashboards) | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/calls |
| Health check | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/health |
| Dispatcher Prometheus metrics | http://k8s-dispatch-dispatch-391c474ddf-1405760084.eu-west-1.elb.amazonaws.com/metrics |

Quick checks:

- **Verify the stack is up:** open the **Health** URL above; you should get a successful JSON response from the dispatcher.
- **See live data:** open **List calls** or any **dashboard** link; dashboards consume the same backend as `/calls`.

### Monitoring

| Tool | URL |
|------|-----|
| Grafana | http://k8s-monitori-grafanai-ad04562758-1939169924.eu-west-1.elb.amazonaws.com |

### Webhooks (integrations)

These endpoints are for **external systems** (voice pipeline, Meta WhatsApp, etc.), not for casual browsing. They are recorded here as the deployed integration URLs from `links.txt`:

| Channel | URL |
|---------|-----|
| Voice / ElevenLabs (example) | https://preauricular-willian-clapped.ngrok-free.dev/webhook |
| WhatsApp | https://preauricular-willian-clapped.ngrok-free.dev/webhooks/whatsapp |

**Note:** Tunnel URLs (for example **ngrok**) often change when the tunnel process or account configuration changes. If webhooks stop receiving traffic, confirm the current public URL in your deployment configuration and update `links.txt` for documentation.

---

## API reference (local shape, same paths in deployment)

- Dispatcher: [`docs/openapi-dispatcher.yaml`](docs/openapi-dispatcher.yaml)
- Classifier: [`docs/openapi-classifier.yaml`](docs/openapi-classifier.yaml)

---

## Repository layout (high level)

| Area | Role |
|------|------|
| `classifier-api/` | Flask service: `/api/dispatch`, `/health`, `/metrics` |
| `dispatcher-api/` | Node service: webhooks, GPT enrichment, DB, `/calls` |
| `dashboard/` | Static UI served with nginx (proxies API paths in Docker) |
| `whatsapp-bot/` | WhatsApp integration layer |
| `k8s/`, `terraform/` | Kubernetes and AWS infrastructure |

For a concise list of every deployed link in one place, see **[`links.txt`](links.txt)**.
