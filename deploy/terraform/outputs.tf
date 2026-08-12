output "app_url" {
  value = google_cloud_run_v2_service.app.uri
}

output "dispatcher_url" {
  value = google_cloudfunctions2_function.dispatcher.service_config[0].uri
}

output "artifact_bucket" {
  value = google_storage_bucket.artifacts.name
}

output "app_service_account" {
  value = google_service_account.app.email
}

output "worker_service_account" {
  value = google_service_account.worker.email
}
