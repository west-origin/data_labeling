# ADR 0001: 개발 도구 체인과 개발 서비스 구성

- 상태: 채택
- 날짜: 2026-10-07
- 관련: WP0

## 결정

1. **Python 도구 체인을 하나로 고정한다.** Python 3.12, uv 워크스페이스(`packages/*`), ruff(린트·포맷), pyright strict, pytest. 모든 패키지를 `make check` 하나로 검증한다.
2. **객체 저장소는 SeaweedFS(Apache 2.0)를 기본으로 한다.** MinIO는 AGPL-3.0이고 커뮤니티판 배포 정책이 바뀌어 기본안에서 뺐다. S3 API만 쓰므로 `config/defaults.yaml`의 값만 바꿔 교체할 수 있다.
3. **원본과 라벨링 데이터를 버킷 단위로 분리한다.** `dlp-raw`(원본), `dlp-labeling`(블러본·프록시), `dlp-datasets`, `dlp-mlflow`. 권한 분리는 WP6·WP16에서 버킷 정책으로 건다.
4. **CVAT는 공식 compose를 고정 버전으로 따로 띄운다.** CVAT는 10개 이상의 서비스로 구성되고 자주 바뀌므로, 우리 compose에 복제하지 않고 `services/cvat/cvat.sh`가 지정 태그를 받아 실행한다. CI 스모크 테스트에서는 이미지 크기 때문에 제외한다.
5. **MLflow 이미지는 직접 빌드한다.** 공식 이미지에는 PostgreSQL·S3 드라이버가 없다.
6. **Label Studio 호스트 포트는 8081이다.** CVAT가 8080을 쓰기 때문이다.

## 결과

- 개발 서비스는 `make up` 한 번으로 기동하고 `make health`로 확인한다.
- 미결정 사항은 `config/defaults.yaml`의 기본값으로 진행하고, 결정이 나면 값만 바꾼다.
