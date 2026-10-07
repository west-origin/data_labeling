#!/bin/sh
# 환경 변수로 S3 자격 증명 파일을 만든 뒤 (관리자, 라벨러: 라벨링 버킷 읽기 전용) 단일 노드 SeaweedFS(마스터+볼륨+파일러+S3)를 띄운다.
# 입력 환경 변수 (services/docker-compose.yml seaweedfs.environment):
#   S3_ACCESS_KEY / S3_SECRET_KEY                — 서비스 계정 (모든 버킷 Admin·Read·List·Tagging·Write)
#   S3_LABELER_ACCESS_KEY / S3_LABELER_SECRET_KEY — 일반 라벨러 (LABELING_BUCKET 읽기·목록만)
#   LABELING_BUCKET                              — 라벨러가 읽을 수 있는 유일한 버킷 (dlp-labeling)
#   VOLUME_SIZE_LIMIT_MB, VOLUME_MAX             — 선택. 볼륨 크기(MB, 기본 1024)·최대 개수(기본 200)
# 라벨러 키로는 원본 버킷(dlp-raw)을 읽을 수 없다 (WP6 역할 경계, 서명 URL도 이 키로 만든다).
# 자격 증명 파일은 컨테이너의 /tmp에만 두고 기동할 때마다 다시 만든다.
set -eu
cat > /tmp/s3.json <<JSON
{
  "identities": [
    {
      "name": "dlp",
      "credentials": [{"accessKey": "${S3_ACCESS_KEY}", "secretKey": "${S3_SECRET_KEY}"}],
      "actions": ["Admin", "Read", "List", "Tagging", "Write"]
    },
    {
      "name": "labeler",
      "credentials": [{"accessKey": "${S3_LABELER_ACCESS_KEY}", "secretKey": "${S3_LABELER_SECRET_KEY}"}],
      "actions": ["Read:${LABELING_BUCKET}", "List:${LABELING_BUCKET}"]
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
