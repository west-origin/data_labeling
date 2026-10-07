#!/bin/sh
# 환경 변수로 S3 자격 증명 파일을 만든 뒤 단일 노드 SeaweedFS(마스터+볼륨+파일러+S3)를 띄운다.
set -eu
cat > /tmp/s3.json <<JSON
{
  "identities": [
    {
      "name": "dlp",
      "credentials": [{"accessKey": "${S3_ACCESS_KEY}", "secretKey": "${S3_SECRET_KEY}"}],
      "actions": ["Admin", "Read", "List", "Tagging", "Write"]
    }
  ]
}
JSON
exec weed server -dir=/data -ip=seaweedfs -ip.bind=0.0.0.0 -s3 -s3.port=8333 -s3.config=/tmp/s3.json
