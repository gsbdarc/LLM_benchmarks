locals {
  app_name    = "pdf-harness"
  worker_name = "pdf-harness-worker"
}

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com", "artifactregistry.googleapis.com", "secretmanager.googleapis.com",
    "storage.googleapis.com", "cloudfunctions.googleapis.com", "cloudbuild.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

resource "google_storage_bucket" "artifacts" {
  name                        = var.artifact_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  versioning {
    enabled = true
  }
}

resource "google_service_account" "app" {
  account_id   = "pdf-harness-app"
  display_name = "PDF Harness Streamlit"
}

resource "google_service_account" "worker" {
  account_id   = "pdf-harness-worker"
  display_name = "PDF Harness worker"
}

resource "google_service_account" "dispatcher" {
  account_id   = "pdf-harness-dispatch"
  display_name = "PDF Harness dispatcher"
}

resource "google_secret_manager_secret_iam_member" "app_password" {
  secret_id = var.app_password_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "worker_mongo" {
  secret_id = var.mongo_uri_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_secret_manager_secret_iam_member" "app_mongo" {
  secret_id = var.mongo_uri_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "dispatcher_mongo" {
  secret_id = var.mongo_uri_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.dispatcher.email}"
}

resource "google_storage_bucket_iam_member" "app_bucket" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.app.email}"
}

resource "google_storage_bucket_iam_member" "worker_bucket" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_storage_bucket_iam_member" "app_bucket_metadata" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.bucketViewer"
  member = "serviceAccount:${google_service_account.app.email}"
}

resource "google_storage_bucket_iam_member" "worker_bucket_metadata" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.bucketViewer"
  member = "serviceAccount:${google_service_account.worker.email}"
}

data "archive_file" "dispatcher" {
  type        = "zip"
  source_dir  = "${path.module}/../../cloud_function"
  output_path = "${path.module}/.terraform/pdf-harness-dispatcher.zip"
  excludes    = ["__pycache__", "__pycache__/*", "*.pyc"]
}

resource "google_storage_bucket_object" "dispatcher_source" {
  name   = "deploy/dispatcher-${data.archive_file.dispatcher.output_md5}.zip"
  bucket = google_storage_bucket.artifacts.name
  source = data.archive_file.dispatcher.output_path
}

resource "google_cloudfunctions2_function" "dispatcher" {
  name     = "pdf-harness-dispatch"
  location = var.region

  build_config {
    runtime     = "python312"
    entry_point = "dispatch_run"
    source {
      storage_source {
        bucket = google_storage_bucket.artifacts.name
        object = google_storage_bucket_object.dispatcher_source.name
      }
    }
  }

  service_config {
    available_memory      = "512M"
    timeout_seconds       = 60
    service_account_email = google_service_account.dispatcher.email
    environment_variables = {
      HARNESS_GCP_PROJECT   = var.project_id
      HARNESS_GCP_REGION    = var.region
      HARNESS_CLOUD_RUN_JOB = local.worker_name
      HARNESS_MONGO_DB      = var.mongo_database
    }
    secret_environment_variables {
      key        = "HARNESS_MONGO_URI"
      project_id = var.project_id
      secret     = var.mongo_uri_secret
      version    = "latest"
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_iam_member" "app_llm" {
  for_each  = var.llm_secret_ids
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.app.email}"
}

resource "google_secret_manager_secret_iam_member" "worker_llm" {
  for_each  = var.llm_secret_ids
  secret_id = each.value
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_secret_manager_secret_iam_member" "worker_wandb" {
  count     = var.wandb_api_key_secret == "" ? 0 : 1
  secret_id = var.wandb_api_key_secret
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.worker.email}"
}

resource "google_cloud_run_v2_job" "worker" {
  name     = local.worker_name
  location = var.region
  template {
    template {
      service_account = google_service_account.worker.email
      timeout         = var.worker_timeout
      max_retries     = 2
      containers {
        image   = var.worker_image
        command = ["python", "-m", "pdf_harness.worker"]
        env {
          name  = "HARNESS_ENVIRONMENT"
          value = "production"
        }
        env {
          name  = "HARNESS_ARTIFACT_BACKEND"
          value = "gcs"
        }
        env {
          name  = "HARNESS_GCS_BUCKET"
          value = google_storage_bucket.artifacts.name
        }
        env {
          name  = "HARNESS_GCP_PROJECT"
          value = var.project_id
        }
        env {
          name  = "HARNESS_GCP_REGION"
          value = var.region
        }
        env {
          name  = "HARNESS_CLOUD_RUN_JOB"
          value = local.worker_name
        }
        env {
          name  = "HARNESS_MONGO_DB"
          value = var.mongo_database
        }
        env {
          name  = "HARNESS_LLM_CONNECTORS_JSON"
          value = var.llm_connectors_json
        }
        env {
          name  = "HARNESS_CODE_VERSION"
          value = var.code_version
        }
        env {
          name  = "HARNESS_WEAVE_PROJECT"
          value = var.weave_project
        }
        dynamic "env" {
          for_each = var.wandb_api_key_secret == "" ? [] : [var.wandb_api_key_secret]
          content {
            name = "WANDB_API_KEY"
            value_source {
              secret_key_ref {
                secret  = env.value
                version = "latest"
              }
            }
          }
        }
        env {
          name = "HARNESS_MONGO_URI"
          value_source {
            secret_key_ref {
              secret  = var.mongo_uri_secret
              version = "latest"
            }
          }
        }
        resources {
          limits = {
            cpu    = "2"
            memory = "4Gi"
          }
        }
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service" "app" {
  name     = local.app_name
  location = var.region
  template {
    service_account = google_service_account.app.email
    containers {
      image = var.app_image
      env {
        name  = "HARNESS_ENVIRONMENT"
        value = "production"
      }
      env {
        name  = "HARNESS_ARTIFACT_BACKEND"
        value = "gcs"
      }
      env {
        name  = "HARNESS_GCS_BUCKET"
        value = google_storage_bucket.artifacts.name
      }
      env {
        name  = "HARNESS_GCP_PROJECT"
        value = var.project_id
      }
      env {
        name  = "HARNESS_GCP_REGION"
        value = var.region
      }
      env {
        name  = "HARNESS_CLOUD_RUN_JOB"
        value = google_cloud_run_v2_job.worker.name
      }
      env {
        name  = "HARNESS_DISPATCH_URL"
        value = google_cloudfunctions2_function.dispatcher.service_config[0].uri
      }
      env {
        name  = "HARNESS_DISPATCH_AUDIENCE"
        value = google_cloudfunctions2_function.dispatcher.service_config[0].uri
      }
      env {
        name  = "HARNESS_APP_PASSWORD_SECRET"
        value = "projects/${var.project_id}/secrets/${var.app_password_secret}/versions/latest"
      }
      env {
        name  = "HARNESS_MONGO_DB"
        value = var.mongo_database
      }
      env {
        name  = "HARNESS_LLM_CONNECTORS_JSON"
        value = var.llm_connectors_json
      }
      env {
        name  = "HARNESS_CODE_VERSION"
        value = var.code_version
      }
      env {
        name  = "HARNESS_MAX_UPLOAD_MB"
        value = tostring(var.max_upload_mb)
      }
      env {
        name  = "STREAMLIT_SERVER_MAX_UPLOAD_SIZE"
        value = tostring(var.max_upload_mb)
      }
      env {
        name = "HARNESS_MONGO_URI"
        value_source {
          secret_key_ref {
            secret  = var.mongo_uri_secret
            version = "latest"
          }
        }
      }
      resources {
        limits = {
          cpu    = "2"
          memory = "2Gi"
        }
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_job_iam_member" "dispatcher_invokes_job" {
  name     = google_cloud_run_v2_job.worker.name
  location = var.region
  role     = "roles/run.jobsExecutorWithOverrides"
  member   = "serviceAccount:${google_service_account.dispatcher.email}"
}

resource "google_cloud_run_v2_job_iam_member" "app_views_job" {
  name     = google_cloud_run_v2_job.worker.name
  location = var.region
  role     = "roles/run.viewer"
  member   = "serviceAccount:${google_service_account.app.email}"
}

resource "google_cloud_run_service_iam_member" "app_invokes_dispatcher" {
  service  = google_cloudfunctions2_function.dispatcher.name
  location = google_cloudfunctions2_function.dispatcher.location
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.app.email}"
}

resource "google_cloud_run_v2_service_iam_member" "app_invokers" {
  for_each = var.app_invoker_members
  name     = google_cloud_run_v2_service.app.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}
