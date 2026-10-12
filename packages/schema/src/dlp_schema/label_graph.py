"""현재 라벨의 상위 구간 연결 검사. DB 저장과 에피소드 계약이 같은 규칙을 사용한다.

노드 식별 (F-SCH-01)
    노드 ID는 label_id가 아니라 action_id / segment_id다. 양손 행동은 두 손 레코드가 같은
    action_id를 쓰므로(ADR 0002 §3) 같은 action_id·다른 손 레코드는 한 노드로 묶는다. 같은 손에
    같은 action_id가 겹치거나, segment_id가 다른 구간·행동 ID와 겹치면 자식 참조가 모호해 위반이다.
"""

from collections.abc import Sequence

from dlp_schema.labels import ActionPayload, LabelRecord, SegmentLevel, SegmentPayload

# 상위 구간 수준의 서열. 자식은 자신보다 낮은 수준이어야 한다 (행동은 모든 구간보다 낮다).
_RANKS = {SegmentLevel.SKILL: 1, SegmentLevel.SUBSTEP: 2, SegmentLevel.TASK: 3}


def _collect_nodes(labels: Sequence[LabelRecord]) -> tuple[dict[str, list[LabelRecord]], list[str]]:
    """노드 ID → 그 노드의 레코드 목록과 ID 중복 위반을 만든다.

    행동 노드는 손마다 레코드 하나라 양손이면 레코드가 둘이다. 구간 노드는 레코드 하나다.

    Returns:
        (노드 색인, 위반 메시지). 중복된 레코드도 색인에 남는다 (자식 검사는 모든 레코드를 본다).
    """
    nodes: dict[str, list[LabelRecord]] = {}
    problems: list[str] = []
    for label in labels:
        p = label.payload
        if isinstance(p, ActionPayload):
            group = nodes.setdefault(p.action_id, [])
            hands = {x.payload.hand for x in group if isinstance(x.payload, ActionPayload)}
            if any(isinstance(x.payload, SegmentPayload) for x in group):
                problems.append(f"구간·행동 ID 중복: {p.action_id}")
            elif p.hand in hands:
                problems.append(f"구간·행동 ID 중복: {p.action_id} (같은 손 {p.hand.value})")
            group.append(label)
        elif isinstance(p, SegmentPayload):
            group = nodes.setdefault(p.segment_id, [])
            if group:
                problems.append(f"구간·행동 ID 중복: {p.segment_id}")
            group.append(label)
    return nodes, problems


def _check_children(
    label: LabelRecord, p: SegmentPayload, nodes: dict[str, list[LabelRecord]]
) -> list[str]:
    """구간 하나의 자식 ID 중복·존재·계층 순서·시간 포함을 검사한다.

    자식이 양손 행동이면 두 손 레코드가 모두 상위 구간 안에 있어야 한다.
    """
    ref = p.segment_id
    problems: list[str] = []
    if len(set(p.child_ids)) != len(p.child_ids):
        problems.append(f"{ref}: 자식 ID 중복")
    for child_id in p.child_ids:
        group = nodes.get(child_id)
        if not group:
            problems.append(f"{ref}: 알 수 없는 자식 구간·행동 {child_id}")
            continue
        if any(
            isinstance(cp := child.payload, SegmentPayload) and _RANKS[cp.level] >= _RANKS[p.level]
            for child in group
        ):
            problems.append(f"{ref}: 자식 {child_id}의 계층 순서 위반")
        if any(
            child.t_start_ms < label.t_start_ms or child.t_end_ms > label.t_end_ms
            for child in group
        ):
            problems.append(f"{ref}: 자식 {child_id}이 상위 구간 밖에 있습니다")
    return problems


def _find_cycles(edges: dict[str, tuple[str, ...]]) -> list[str]:
    """구간 참조 순환을 찾는다. 반복 DFS로 긴 계층에서도 Python 재귀 한도를 피한다."""
    problems: list[str] = []
    finished: set[str] = set()
    for root in edges:
        visiting: set[str] = set()
        stack = [(root, False)]
        while stack:
            node, leaving = stack.pop()
            if leaving:
                visiting.discard(node)
                finished.add(node)
            elif node in visiting:
                problems.append(f"구간 참조 순환: {node}")
            elif node not in finished:
                visiting.add(node)
                stack.append((node, True))
                stack.extend((child, False) for child in edges.get(node, ()) if child in edges)
    return problems


def check_segment_links(labels: Sequence[LabelRecord]) -> list[str]:
    """노드 ID 중복과 자식 ID·중복·시간 포함·계층 순서·순환을 검사한다. 입력은 현재 라벨이어야 한다.

    ID는 label_id가 아니라 action_id / segment_id다. 같은 action_id·다른 손 레코드(양손 행동)는
    한 노드이고, 같은 손 중복만 위반이다. 같은 수준이나 상위 수준을 자식으로 묶을 수 없다.
    하위 단계가 없는 파일럿도 표현하도록 수준을 건너뛰는 연결은 허용한다.

    Returns:
        위반 메시지 목록 (없으면 빈 목록).
    """
    nodes, problems = _collect_nodes(labels)
    edges: dict[str, tuple[str, ...]] = {}
    for ref, group in nodes.items():
        for label in group:
            p = label.payload
            if isinstance(p, SegmentPayload):
                # 같은 ID 구간이 여럿이면(이미 중복 위반) 마지막 것의 자식으로 순환을 본다
                edges[ref] = p.child_ids
                problems.extend(_check_children(label, p, nodes))
    problems.extend(_find_cycles(edges))
    return problems
