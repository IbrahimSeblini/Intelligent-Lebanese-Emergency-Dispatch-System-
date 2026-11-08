output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  description = "RDS hostname to put in RDS_HOST"
  value       = aws_db_instance.main.address
}

output "ecr_urls" {
  description = "ECR repository URLs for docker push"
  value       = { for k, v in aws_ecr_repository.services : k => v.repository_url }
}

output "configure_kubectl" {
  description = "Run this after apply to connect kubectl to the cluster"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
