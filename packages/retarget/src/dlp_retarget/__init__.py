"""사람 손 3D → 로봇 관절 궤적 (리타기팅, ADR 0047).

고객이 준 로봇 설계(URDF)에 사람 1인칭 손 동작(세션 월드 좌표 손목·엄지 끝·검지 끝, ADR 0046)을 옮겨
로봇 관절 각 궤적과 프레임별 도달 가능성을 낸다. 내보내기(`dlp export lerobot --robot <ID>`,
`dlp_export.robot`)가 부른다. 영상은 바꾸지 않는다.

모듈:
- `urdf`: URDF 관절만 읽는 직렬 연쇄 정기구학·야코비안.
- `robot`: 로봇 등록(`config/robots/<ID>.yaml`), 모델 파일 sha256 확인, 손 → 로봇 배치·그리퍼 설정.
- `policy`: 리타기팅 정책(`config/policies/retarget.yaml`).
- `targets`: 손 점 → 끝단 목표(위치·접근 축·닫힘 축)와 그리퍼 열림.
- `ik`: IK 풀이기 인터페이스와 내장 감쇠 최소제곱 풀이기(관절·속도 한계, 매끄러움).
- `mink`: 격리 환경의 mink(MuJoCo) 풀이기 어댑터 (선택).
- `retarget`: 에피소드 하나를 리타기팅하고 도달 가능성 보고를 만든다.
"""
