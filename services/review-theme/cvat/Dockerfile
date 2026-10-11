# aniatman 검수 화면 테마를 얹은 CVAT UI 이미지 (ADR 0038).
# 공식 cvat/ui 이미지의 정적 파일은 그대로 두고, /dlp/ 아래에 테마 파일을 넣은 뒤 index.html의
# </head> 앞에 CSS·JS 연결을 덧붙인다. 화면 소스(React)는 고치지 않는다.
# 빌드: services/cvat/cvat.sh up이 services/review-theme을 컨텍스트로 빌드한다.
ARG CVAT_VERSION
FROM cvat/ui:${CVAT_VERSION}

# /dlp/는 nginx의 location /(캐시 안 함)로 나가므로 테마를 바꾸면 새로 고침만으로 반영된다
COPY logo.png /usr/share/nginx/html/dlp/logo.png
COPY cvat/dlp-tokens.css cvat/dlp-cvat.css cvat/dlp-cvat.js /usr/share/nginx/html/dlp/

# 공식 이미지는 nginx 사용자로 돌고 index.html은 root 소유라, 고칠 때만 root로 바꿨다가 되돌린다
USER root
# 연결이 이미 있으면 다시 넣지 않는다. 연결 문자열이 들어가지 않았으면(</head>가 없으면) 빌드를 실패시킨다
RUN f=/usr/share/nginx/html/index.html \
 && if ! grep -q '/dlp/dlp-cvat.js' "$f"; then \
      sed -i 's#</head>#<link rel="stylesheet" href="/dlp/dlp-tokens.css"><link rel="stylesheet" href="/dlp/dlp-cvat.css"><script src="/dlp/dlp-cvat.js"></script></head>#' "$f"; \
    fi \
 && grep -q '/dlp/dlp-cvat.js' "$f"
USER nginx
