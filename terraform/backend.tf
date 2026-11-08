terraform {
  backend "s3" {
    bucket         = "dispatcher-terraform-state-838054321169"
    key            = "dispatcher/terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "dispatcher-terraform-locks"
    encrypt        = true
  }
}
