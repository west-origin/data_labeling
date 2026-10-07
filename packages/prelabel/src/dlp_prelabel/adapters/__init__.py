"""Predictor 어댑터.

실제 모델(MediaPipe, RTMPose, OWLv2)과 CPU용 stub이 같은 인터페이스를 따른다.

인터페이스(`dlp_schema.predictor.Predictor`): 속성 `name`(라벨 ID 접두사이자 러너의 단계 이름),
`version`(모델 출처 라벨의 `model_version`), 메서드 `run(Clip) -> list[LabelRecord]`. `version`에는
가중치 해시(`dlp_models.registry.resolve`)와 그 어댑터가 쓰는 정책 절 해시
(`PrelabelPolicy.digest`)를 넣는다. 버전이 바뀌면 러너가 다시 돌리고 검수 전인 이전 결과를 지운다.

- `mediapipe_models`: `MediaPipeHands`(손 21관절, hand21), `MediaPipeObjects`(COCO 객체 박스).
- `rtmpose`: `RtmPose`(YOLOX 사람 탐지 → RTMPose 전신 coco17).
- `owl_objects`: `OwlObjects`(OWLv2 오픈 보캐뷸러리 도구 박스).
- `stubs`: `OraclePredictor`(정답 재출력, CI용), `UnavailablePredictor`(미연동 기능 자리).

라벨 ID 형식: `<세션>-<스트림>-<name>-<version_tag(version)>-<순번 등>`. 러너는
`<세션>-<스트림>-<name>-` 접두사로 같은 어댑터의 이전 버전을 찾으므로 이 형식을 지켜야 한다.
"""
