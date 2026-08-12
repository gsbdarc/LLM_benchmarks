# Google Cloud deployment and credentials

The app container, worker container, and Gen-2 dispatcher function use the same Mongo
database but separate service identities. PDFs and rendered pages live in a private GCS
bucket. Mongo holds URIs and metadata only.

## Credentials to supply

Do not paste values into chat, Terraform variables, source files, or Mongo documents.

1. `pdf-harness-mongo-uri`: the complete URI for the new harness database user. Give this
   user access only to the new database. Add the GCP egress address to the Atlas allowlist.
2. `pdf-harness-app-password`: a strong temporary shared password for the internal UI.
3. One Secret Manager secret per LLM API credential. Bind it to an approved HTTPS
   connector and allowed model IDs in `HARNESS_LLM_CONNECTORS_JSON`; users only select
   connector/model IDs and cannot enter URLs or secret references.
4. No downloaded GCP service-account JSON key is needed. Cloud Run uses attached service
   accounts for GCS, Secret Manager, function invocation, and job execution.
5. Optional W&B Weave: bind `WANDB_API_KEY` as a worker secret environment variable and
   set Terraform variables `wandb_api_key_secret` and `weave_project`. Only sanitized
   metadata is mirrored; Mongo is authoritative.

Create secret containers and add values interactively:

```bash
gcloud secrets create pdf-harness-mongo-uri --replication-policy=automatic
gcloud secrets versions add pdf-harness-mongo-uri --data-file=-
gcloud secrets create pdf-harness-app-password --replication-policy=automatic
gcloud secrets versions add pdf-harness-app-password --data-file=-
gcloud secrets create stanford-ai-api-key --replication-policy=automatic
gcloud secrets versions add stanford-ai-api-key --data-file=-
```

`--data-file=-` reads the value without placing it in shell history. Grant the worker and
app service accounts access only to the LLM secrets used by their projects.

## Build and deploy order

1. Build one container image and pin its digest in `app_image` and `worker_image`.
   For an existing Artifact Registry Docker repository:

   ```bash
   gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT/REPOSITORY/pdf-harness:VERSION .
   gcloud artifacts docker images describe REGION-docker.pkg.dev/PROJECT/REPOSITORY/pdf-harness:VERSION --format='value(image_summary.digest)'
   ```

   Copy `deploy/terraform/terraform.tfvars.example` to an untracked `.tfvars` file and
   replace the image tags with the returned immutable digest.
2. Create the two secret containers above; Terraform deliberately does not manage values.
3. Apply `deploy/terraform`. Terraform packages and deploys the private Gen-2 dispatcher,
   creates the bucket, identities, Streamlit service, worker job, and their IAM bindings in
   one graph. Supply `app_invoker_members` with approved users/groups who may reach the
   internal password screen.
4. Terraform grants the dispatcher identity permission to execute the worker with run-ID
   overrides, and grants the Streamlit identity permission to invoke the function.
5. Keep the Streamlit Cloud Run service IAM-restricted during the password-prototype phase.

The Terraform is intentionally a deployment baseline: environment-specific networking,
Atlas static egress, domain mapping, retention, alert policies, and Stanford SSO/IAP must
be supplied by the owning GCP project before production exposure.
