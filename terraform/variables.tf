variable "region" {
  description = "AWS region"
  default     = "eu-west-1"
}

variable "project" {
  description = "Project name used as prefix for all resources"
  default     = "dispatcher"
}

variable "db_password" {
  description = "RDS master password"
  sensitive   = true
}

variable "eks_node_instance_type" {
  description = "EC2 instance type for EKS worker nodes"
  default     = "t3.small"
}

variable "eks_node_count" {
  description = "Desired number of EKS worker nodes"
  default     = 2
}
