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
# 버킷마다 별도 볼륨(컬렉션)을 쓴다. 기본 볼륨 크기(30GB)로는 디스크 여유 공간에 따라 볼륨이
# 한두 개만 잡혀 두 번째 버킷부터 쓰기가 실패하므로, 개발용으로 작게 잡고 개수를 늘린다.
exec weed server -dir=/data -ip=seaweedfs -ip.bind=0.0.0.0 \
  -master.volumeSizeLimitMB="${VOLUME_SIZE_LIMIT_MB:-1024}" -master.volumePreallocate=false \
  -volume.max="${VOLUME_MAX:-200}" \
  -s3 -s3.port=8333 -s3.config=/tmp/s3.json
