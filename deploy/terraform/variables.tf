variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-west1"
}

variable "app_image" {
  type        = string
  description = "Artifact Registry image digest for Streamlit"
}

variable "worker_image" {
  type        = string
  description = "Artifact Registry image digest for the worker"
}

variable "artifact_bucket" {
  type = string
}

variable "mongo_database" {
  type    = string
  default = "pdf_extraction_harness"
}

variable "mongo_uri_secret" {
  type    = string
  default = "pdf-harness-mongo-uri"
}

variable "app_password_secret" {
  type    = string
  default = "pdf-harness-app-password"
}

variable "llm_connectors_json" {
  type        = string
  sensitive   = true
  description = "Admin-approved connector metadata and Secret Manager refs; contains no secret values"
}

variable "llm_secret_ids" {
  type        = set(string)
  default     = []
  description = "Secret IDs referenced by llm_connectors_json"
}

variable "app_invoker_members" {
  type        = set(string)
  default     = []
  description = "IAM members allowed to reach the internal Streamlit service"
}

variable "code_version" {
  type        = string
  description = "Git SHA or immutable image version stamped onto run snapshots"
}

variable "worker_timeout" {
  type    = string
  default = "86400s"
}

variable "max_upload_mb" {
  type    = number
  default = 200
}

variable "weave_project" {
  type    = string
  default = ""
}

variable "wandb_api_key_secret" {
  type    = string
  default = ""
}
