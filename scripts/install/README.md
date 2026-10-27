# 플랫폼 설치 (Mac · Linux · Windows WSL2)

`install.sh`는 저장소에 이미 있는 Make 대상(`make install`, `make up`, `make db-upgrade`, `make models`,
`make cvat-up`, `make cvat-superuser`)을 순서대로 부르는 얇은 스크립트다. 단계마다 다시 돌려도 안전하고,
실패하면 할 일을 한국어로 찍고 멈춘다. 설치가 끝나면 `uv run dlp console serve`로 운영 콘솔을 연다
(ADR 0061, 안내서 [operator-guide.md](../../docs/handover/operator-guide.md)).

## 필요한 것

| 항목 | 권장 | 비고 |
| --- | --- | --- |
| 디스크 | 60 GB 이상 | 서비스 이미지·볼륨·모델(약 0.6 GB)·영상. CVAT를 띄우면 더 든다 |
| 메모리 | 16 GB 이상 | 긴 영상 프리라벨은 단계를 나눠 돌린다 |
| Docker | Docker Desktop(Mac·Windows) 또는 Docker Engine + Compose v2(Linux) | |
| uv | 0.11 이상 | 없으면 `--install-uv` |
| GPU | 선택 | NVIDIA GPU + nvidia-container-toolkit가 있으면 감지해 알린다. 개발 서비스 자체는 CPU로 돈다 |

## Mac · Linux

```bash
git clone <저장소 주소> data_labeling && cd data_labeling
scripts/install/install.sh --check-only     # 점검만
scripts/install/install.sh --install-uv     # 설치 (CVAT까지: --with-cvat)
uv run dlp console serve
```

Apple Silicon Mac에서는 CVAT 이미지가 amd64라 에뮬레이션으로 돈다 (Docker Desktop의 Rosetta 사용을 켠다).

## Windows (WSL2)

PowerShell 래퍼는 두지 않았다. WSL2 안의 Ubuntu에서 Linux와 같은 방법으로 설치한다.

1. 관리자 PowerShell에서 `wsl --install -d Ubuntu` → 다시 시작 → Ubuntu 사용자 만들기.
2. Docker Desktop 설치 → Settings → Resources → WSL integration에서 Ubuntu 켜기.
3. Ubuntu 터미널에서 **WSL 안의 디렉터리**(예: `~/data_labeling`, `/mnt/c/...`는 느리다)에 저장소를 받고
   위 Mac·Linux 명령을 그대로 실행한다. MediaPipe용 `sudo apt install libegl1 libgles2 ffmpeg make`.
4. 콘솔은 Windows 브라우저에서 `http://localhost:8090/#token=…`으로 열린다 (WSL2가 localhost를 이어 준다).

## 옵션

| 옵션 | 하는 일 |
| --- | --- |
| `--check-only` | 사전 점검만 (설치·실행 없음) |
| `--install-uv` | uv가 없으면 공식 설치 스크립트로 설치 |
| `--curation` | `make install-curation` (FiftyOne 포함, 약 1 GB 더) |
| `--skip-models` | 모델 가중치 받기를 건너뜀 |
| `--with-cvat` | CVAT까지 띄우고 관리자 계정을 만든다 |

`.env`는 없을 때만 `.env.example`에서 만들고 비밀번호·키를 무작위로 바꾼다 (권한 0600, 값은 찍지 않는다).
이미 있으면 그대로 둔다. `dlp` 명령은 `.env`를 스스로 읽지 않으므로 터미널에서 직접 쓸 때는
`set -a; . ./.env; set +a`를 먼저 한다 (콘솔은 스스로 읽어 단계 명령에 넘긴다).
