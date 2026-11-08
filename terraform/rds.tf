resource "aws_db_subnet_group" "main" {
  name       = "${var.project}-db-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "rds" {
  name   = "${var.project}-rds-sg"
  vpc_id = module.vpc.vpc_id

  ingress {
    description = "Postgres from within VPC"
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "main" {
  identifier        = var.project
  engine            = "postgres"
  engine_version    = "16"
  instance_class    = "db.t3.micro"
  allocated_storage = 20
  storage_type      = "gp3"

  db_name  = "dispatcher"
  username = "dispatcher"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  publicly_accessible = false
  skip_final_snapshot = true
}
