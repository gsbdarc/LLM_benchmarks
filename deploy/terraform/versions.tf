terraform {
  required_version = ">= 1.6"
  required_providers {
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.5"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
