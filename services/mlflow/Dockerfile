# syntax=docker/dockerfile:1
# 공식 이미지에는 PostgreSQL·S3 드라이버가 없어 직접 빌드한다.
# TLS를 가로채는 사내 프록시 뒤에서는 CA 묶음을 빌드 시크릿으로 넘긴다:
#   docker build --secret id=extra_ca,src=<ca-bundle.crt> -t dlp-mlflow:3.16.1 services/mlflow
# services/docker-compose.yml의 mlflow 서비스가 `build: ./mlflow`, 이미지 이름 dlp-mlflow:3.16.1로 빌드한다.
# psycopg2-binary: PostgreSQL 백엔드 저장소, boto3: S3(SeaweedFS) 산출물 저장소.
FROM python:3.12-slim
# MLflow 버전. 바꾸면 compose의 image 태그와 dlp_train.tracking(REST API 호출) 호환을 함께 확인한다
ARG MLFLOW_VERSION=3.16.1
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" psycopg2-binary boto3
EXPOSE 5000
