module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.project
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_public_access       = true
  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    workers = {
      instance_types = [var.eks_node_instance_type]
      min_size       = 1
      max_size       = 4
      desired_size   = var.eks_node_count
    }
  }
}
