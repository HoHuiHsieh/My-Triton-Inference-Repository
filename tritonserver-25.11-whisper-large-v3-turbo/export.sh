#!/bin/bash
docker save tritonserver:25.11-transformers | gzip > tritonserver-25.11-transformers.tar.gz
