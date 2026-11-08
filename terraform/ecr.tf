locals {
  services = ["classifier-api", "dispatcher-api", "dashboard"]
}

resource "aws_ecr_repository" "services" {
  for_each = toset(local.services)

  name                 = each.key
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}
