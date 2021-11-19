variable "BASE_IMAGE_NAME" {
  default = "pytorch/pytorch"
}

variable "BASE_IMAGE_TAG" {
 default = "1.5.1-cuda10.1-cudnn7-runtime"
}

variable "IMAGE_NAME" {
  default = "HotThoughts/cloudpose_pt"
}

group "default" {
  targets = ["cpu"]
}


target "cpu" {
  dockerfile = "Dockerfile"
  tags = ["${IMAGE_NAME}:latest"]

  args = [
    BASE_IMAGE_NAME = ${BASE_IMAGE_NAME}
    BASE_IMAGE_TAG = ${BASE_IMAGE_TAG}
  ]

  cache-to = ["type=inline"]

  cache-from = [
    "type=registry,ref=${IMAGE_NAME}:latest",
  ]
}


target "gpu" {
  dockerfile = "Dockerfile"
  tags = []
}
