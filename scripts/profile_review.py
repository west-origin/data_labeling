"""합성 PostgreSQL 검수 계획의 조회 비용 측정. 일회용 DB를 만들고 반드시 제거한다.

예: uv run python scripts/profile_review.py --sessions 300 --out reports/review-profile.json
실제 영상·모델 추론 성능이 아니라 DB 검수 운영 경로의 시간·쿼리 수를 측정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import sqlalchemy as sa

from dlp_review.ops.policy import load_policy
from dlp_review.ops.runner import _open_loads, known_classes, plan_session
from dlp_schema import repo_root
from dlp_schema.db import repository as repo
from dlp_schema.db.migrate import upgrade
from dlp_schema.labels import Provenance, Source
from dlp_schema.ontology import load_ontology
from dlp_schema.review import ReviewAssignment, ReviewMode
from dlp_schema.testing import FIXED_TIME, make_label, make_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.sessions < 1 or args.repeats < 1:
        parser.error("sessions와 repeats는 양수여야 합니다")
    url = sa.make_url(os.environ["DLP_DATABASE_URL"])
    name = "dlp_profile_" + uuid.uuid4().hex[:8]
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT")
    engine = sa.create_engine(url.set(database=name))
    query_count = 0

    def count(*_args: object) -> None:
        nonlocal query_count
        query_count += 1

    sa.event.listen(engine, "before_cursor_execute", count)
    try:
        with admin.connect() as conn:
            conn.execute(sa.text(f'CREATE DATABASE "{name}"'))
        upgrade(engine.url.render_as_string(hide_password=False))
        ontology = load_ontology(repo_root() / "config/ontology/v1")
        policy = load_policy(repo_root())
        with engine.begin() as conn:
            repo.register_ontology(conn, ontology)
            for i in range(args.sessions):
                sid = f"profile-{i}"
                repo.insert_session(conn, make_session(session_id=sid))
                labels = [
                    make_label(
                        {
                            "kind": "box_track",
                            "entity_id": f"cup_{j}",
                            "class_id": "cup" if j % 2 else "rag",
                            "keyframes": [
                                {"t_ms": k * 100, "x": 10, "y": 10, "w": 10, "h": 10}
                                for k in range(50)
                            ],
                        },
                        label_id=f"{sid}-box-{j}",
                        session_id=sid,
                        stream_id="bodycam",
                        t_end_ms=4900,
                        provenance=Provenance(source=Source.HUMAN),
                    )
                    for j in range(10)
                ]
                repo.insert_labels(conn, labels)
                for j in range(8):
                    repo.insert_assignment(
                        conn,
                        ReviewAssignment(
                            assignment_id=f"{sid}-a-{j}",
                            session_id=sid,
                            stream_id="bodycam",
                            label_kinds=("box_track",),
                            mode=ReviewMode.STANDARD,
                            priority=1,
                            assignee=f"reviewer-{j % 4}",
                            created_at=FIXED_TIME,
                        ),
                    )
        measurements = {}
        with engine.begin() as conn:
            operations = {
                "known_classes": lambda: known_classes(conn, "profile-0"),
                "open_loads": lambda: _open_loads(conn, [f"reviewer-{j}" for j in range(4)]),
                "plan_session": lambda: plan_session(
                    conn, "profile-0", ["reviewer-0"], policy, ontology, seed=0, now=FIXED_TIME
                ),
            }
            for label, operation in operations.items():
                times, counts = [], []
                operation()  # 접속·쿼리 캐시 워밍업
                for _ in range(args.repeats):
                    query_count = 0
                    start = time.perf_counter()
                    result = operation()
                    times.append((time.perf_counter() - start) * 1000)
                    counts.append(query_count)
                    if label == "known_classes":
                        assert result == {"cup", "rag"}
                measurements[label] = {
                    "median_ms": round(statistics.median(times), 3),
                    "min_ms": round(min(times), 3),
                    "max_ms": round(max(times), 3),
                    "queries": counts,
                }
        result = {
            "sessions": args.sessions,
            "labels": args.sessions * 10,
            "keyframes_per_label": 50,
            "assignments": args.sessions * 8,
            "repeats": args.repeats,
            "measurements": measurements,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        engine.dispose()
        with admin.connect() as conn:
            conn.execute(sa.text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


if __name__ == "__main__":
    main()
