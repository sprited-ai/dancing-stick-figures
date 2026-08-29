#!/usr/bin/env python3
"""Small SQLite-backed review queue for the dataset paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "paper" / "dataset_paper_v3.review.sqlite3"
DEFAULT_SOURCE = "paper/dataset_paper_v3.sqlite-pass.tex"

SCHEMA = """
CREATE TABLE IF NOT EXISTS review_comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_pass TEXT NOT NULL DEFAULT 'v3-sqlite-pass',
    source_file TEXT NOT NULL,
    section TEXT NOT NULL,
    paragraph_anchor TEXT NOT NULL,
    anchor_sha256 TEXT NOT NULL,
    paragraph_sha256 TEXT,
    source_sha256 TEXT,
    reviewer_comment TEXT NOT NULL,
    implementation_instruction TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 0 AND 3),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'claimed', 'implemented', 'verified', 'reopened', 'declined')),
    created_by TEXT NOT NULL DEFAULT 'pixel',
    claimed_by TEXT,
    implementation_summary TEXT,
    implementation_evidence TEXT,
    verifier_note TEXT,
    created_at TEXT NOT NULL,
    claimed_at TEXT,
    lease_expires_at TEXT,
    implemented_at TEXT,
    verified_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_review_comments_queue
ON review_comments(status, priority DESC, id);

CREATE TABLE IF NOT EXISTS review_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id INTEGER NOT NULL REFERENCES review_comments(id),
    actor TEXT NOT NULL,
    event TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    related_claim TEXT NOT NULL,
    evidence_present TEXT NOT NULL,
    missing_evidence TEXT NOT NULL,
    minimum_experiment TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 1 CHECK (priority BETWEEN 0 AND 3),
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK (status IN ('candidate', 'selected', 'running', 'completed', 'deferred', 'not_needed')),
    reviewer_note TEXT,
    created_by TEXT NOT NULL DEFAULT 'pixel',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_gaps_queue
ON experiment_gaps(status, priority DESC, id);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def stamp(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat().replace("+00:00", "Z")


def anchor_hash(anchor: str) -> str:
    normalized = " ".join(anchor.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paragraph_for_anchor(path: Path, anchor: str) -> str:
    text = path.read_text(encoding="utf-8")
    occurrences = text.count(anchor)
    if occurrences != 1:
        raise SystemExit(
            f"anchor must occur exactly once in {path}; found {occurrences}: {anchor!r}"
        )
    pos = text.index(anchor)
    start = text.rfind("\n\n", 0, pos)
    end = text.find("\n\n", pos)
    start = 0 if start < 0 else start + 2
    end = len(text) if end < 0 else end
    return " ".join(text[start:end].split())


def paragraph_hash(path: Path, anchor: str) -> str:
    paragraph = paragraph_for_anchor(path, anchor)
    return hashlib.sha256(paragraph.encode("utf-8")).hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(review_comments)")}
    if "paragraph_sha256" not in columns:
        conn.execute("ALTER TABLE review_comments ADD COLUMN paragraph_sha256 TEXT")
    if "source_sha256" not in columns:
        conn.execute("ALTER TABLE review_comments ADD COLUMN source_sha256 TEXT")
    return conn


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def emit(row: sqlite3.Row | None) -> None:
    print(json.dumps(dict(row) if row else None, ensure_ascii=False, indent=2))


def event(conn: sqlite3.Connection, comment_id: int, actor: str, name: str, note: str = "") -> None:
    conn.execute(
        "INSERT INTO review_events(comment_id, actor, event, note, created_at) VALUES (?, ?, ?, ?, ?)",
        (comment_id, actor, name, note, stamp()),
    )


def cmd_init(conn: sqlite3.Connection, _args: argparse.Namespace) -> None:
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(json.dumps({"ok": True, "journal_mode": mode}, indent=2))


def cmd_add(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    source_path = (ROOT / args.source_file).resolve()
    if not source_path.is_file():
        raise SystemExit(f"source file does not exist: {source_path}")
    para_sha = paragraph_hash(source_path, args.anchor)
    src_sha = file_hash(source_path)
    with immediate(conn):
        cur = conn.execute(
            """
            INSERT INTO review_comments(
                review_pass, source_file, section, paragraph_anchor, anchor_sha256,
                paragraph_sha256, source_sha256,
                reviewer_comment, implementation_instruction, priority, created_by,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.review_pass,
                args.source_file,
                args.section,
                args.anchor,
                anchor_hash(args.anchor),
                para_sha,
                src_sha,
                args.comment,
                args.instruction,
                args.priority,
                args.actor,
                now,
                now,
            ),
        )
        comment_id = int(cur.lastrowid)
        event(conn, comment_id, args.actor, "created", args.comment)
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (comment_id,)).fetchone())


def release_expired_claims(conn: sqlite3.Connection) -> None:
    now = stamp()
    expired = conn.execute(
        "SELECT id, claimed_by FROM review_comments WHERE status = 'claimed' AND lease_expires_at < ?",
        (now,),
    ).fetchall()
    for row in expired:
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'reopened', claimed_by = NULL, claimed_at = NULL,
                lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (now, row["id"]),
        )
        event(conn, row["id"], "queue", "lease_expired", str(row["claimed_by"] or ""))


def cmd_claim(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now_dt = utc_now()
    with immediate(conn):
        release_expired_claims(conn)
        row = conn.execute(
            """
            SELECT * FROM review_comments
            WHERE status IN ('open', 'reopened')
            ORDER BY priority DESC, id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            emit(None)
            return
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'claimed', claimed_by = ?, claimed_at = ?, lease_expires_at = ?, updated_at = ?
            WHERE id = ? AND status IN ('open', 'reopened')
            """,
            (
                args.worker,
                stamp(now_dt),
                stamp(now_dt + timedelta(minutes=args.lease_minutes)),
                stamp(now_dt),
                row["id"],
            ),
        )
        event(conn, row["id"], args.worker, "claimed")
        claimed = conn.execute("SELECT * FROM review_comments WHERE id = ?", (row["id"],)).fetchone()
    emit(claimed)


def cmd_implement(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        row = conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown comment id: {args.id}")
        if row["status"] != "claimed" or row["claimed_by"] != args.worker:
            raise SystemExit("comment must be claimed by this worker before implementation")
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'implemented', implementation_summary = ?, implementation_evidence = ?,
                implemented_at = ?, lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (args.summary, args.evidence, now, now, args.id),
        )
        event(conn, args.id, args.worker, "implemented", args.summary)
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone())


def cmd_verify(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        row = conn.execute("SELECT status FROM review_comments WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown comment id: {args.id}")
        if row["status"] != "implemented":
            raise SystemExit("only an implemented comment can be verified")
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'verified', verifier_note = ?, verified_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (args.note, now, now, args.id),
        )
        event(conn, args.id, args.reviewer, "verified", args.note)
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone())


def cmd_reopen(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        row = conn.execute("SELECT status FROM review_comments WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown comment id: {args.id}")
        if row["status"] not in ("implemented", "claimed"):
            raise SystemExit("only a claimed or implemented comment can be reopened")
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'reopened', claimed_by = NULL, claimed_at = NULL, lease_expires_at = NULL,
                verifier_note = ?, updated_at = ?
            WHERE id = ?
            """,
            (args.reason, now, args.id),
        )
        event(conn, args.id, args.reviewer, "reopened", args.reason)
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone())


def cmd_decline(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        row = conn.execute("SELECT status FROM review_comments WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown comment id: {args.id}")
        if row["status"] == "verified":
            raise SystemExit("a verified comment cannot be declined")
        conn.execute(
            """
            UPDATE review_comments
            SET status = 'declined', verifier_note = ?, claimed_by = NULL,
                lease_expires_at = NULL, updated_at = ?
            WHERE id = ?
            """,
            (args.reason, now, args.id),
        )
        event(conn, args.id, args.reviewer, "declined", args.reason)
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone())


def cmd_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    if args.status == "all":
        rows = conn.execute("SELECT * FROM review_comments ORDER BY id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM review_comments WHERE status = ? ORDER BY priority DESC, id",
            (args.status,),
        ).fetchall()
    print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))


def cmd_show(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    emit(conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone())


def cmd_check(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    row = conn.execute("SELECT * FROM review_comments WHERE id = ?", (args.id,)).fetchone()
    if row is None:
        raise SystemExit(f"unknown comment id: {args.id}")
    source_path = (ROOT / row["source_file"]).resolve()
    if not source_path.is_file():
        print(json.dumps({"id": args.id, "ok": False, "error": "source_missing"}, indent=2))
        return
    try:
        current_para = paragraph_hash(source_path, row["paragraph_anchor"])
    except SystemExit as exc:
        print(json.dumps({"id": args.id, "ok": False, "error": str(exc)}, indent=2))
        return
    current_source = file_hash(source_path)
    print(
        json.dumps(
            {
                "id": args.id,
                "ok": current_para == row["paragraph_sha256"],
                "paragraph_matches": current_para == row["paragraph_sha256"],
                "source_matches_enqueue_time": current_source == row["source_sha256"],
                "current_paragraph_sha256": current_para,
                "queued_paragraph_sha256": row["paragraph_sha256"],
                "current_source_sha256": current_source,
                "queued_source_sha256": row["source_sha256"],
            },
            indent=2,
        )
    )


def cmd_stats(conn: sqlite3.Connection, _args: argparse.Namespace) -> None:
    rows = conn.execute(
        "SELECT status, COUNT(*) AS count FROM review_comments GROUP BY status ORDER BY status"
    ).fetchall()
    print(json.dumps({row["status"]: row["count"] for row in rows}, indent=2))


def cmd_history(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute(
        "SELECT * FROM review_events WHERE comment_id = ? ORDER BY id",
        (args.id,),
    ).fetchall()
    print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))


def cmd_export(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    rows = conn.execute("SELECT * FROM review_comments ORDER BY id").fetchall()
    gaps = conn.execute("SELECT * FROM experiment_gaps ORDER BY priority DESC, id").fetchall()
    counts = conn.execute(
        "SELECT status, COUNT(*) AS count FROM review_comments GROUP BY status ORDER BY status"
    ).fetchall()
    lines = [
        "# Dataset paper SQLite review record",
        "",
        f"Exported: {stamp()}",
        "",
        "## Status",
        "",
    ]
    if counts:
        lines.extend(f"- {row['status']}: {row['count']}" for row in counts)
    else:
        lines.append("- Queue is empty.")
    for row in rows:
        lines.extend(
            [
                "",
                f"## #{row['id']} — {row['section']} [{row['status']}]",
                "",
                f"- Priority: {row['priority']}",
                f"- Source: `{row['source_file']}`",
                f"- Anchor: {row['paragraph_anchor']}",
                f"- Anchor SHA-256: `{row['anchor_sha256']}`",
                f"- Reviewer comment: {row['reviewer_comment']}",
                f"- Implementation instruction: {row['implementation_instruction']}",
            ]
        )
        if row["implementation_summary"]:
            lines.append(f"- Implementation: {row['implementation_summary']}")
        if row["implementation_evidence"]:
            lines.append(f"- Evidence: {row['implementation_evidence']}")
        if row["verifier_note"]:
            lines.append(f"- Pixel verification: {row['verifier_note']}")
    lines.extend(["", "## Experiment-gap ledger", ""])
    if not gaps:
        lines.append("- No experiment gaps recorded.")
    for gap in gaps:
        lines.extend(
            [
                f"### E{gap['id']} — {gap['title']} [{gap['status']}]",
                "",
                f"- Priority: {gap['priority']}",
                f"- Related claim: {gap['related_claim']}",
                f"- Evidence already present: {gap['evidence_present']}",
                f"- Missing evidence: {gap['missing_evidence']}",
                f"- Minimum experiment: {gap['minimum_experiment']}",
            ]
        )
        if gap["reviewer_note"]:
            lines.append(f"- Reviewer note: {gap['reviewer_note']}")
        lines.append("")
    output = "\n".join(lines) + "\n"
    if args.output:
        target = args.output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")
        print(json.dumps({"output": str(target), "comments": len(rows)}, indent=2))
    else:
        print(output, end="")


def cmd_gap_add(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        cur = conn.execute(
            """
            INSERT INTO experiment_gaps(
                title, related_claim, evidence_present, missing_evidence,
                minimum_experiment, priority, created_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.title,
                args.claim,
                args.present,
                args.missing,
                args.minimum,
                args.priority,
                args.actor,
                now,
                now,
            ),
        )
        gap_id = int(cur.lastrowid)
    emit(conn.execute("SELECT * FROM experiment_gaps WHERE id = ?", (gap_id,)).fetchone())


def cmd_gap_list(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    if args.status == "all":
        rows = conn.execute("SELECT * FROM experiment_gaps ORDER BY priority DESC, id").fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM experiment_gaps WHERE status = ? ORDER BY priority DESC, id",
            (args.status,),
        ).fetchall()
    print(json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2))


def cmd_gap_update(conn: sqlite3.Connection, args: argparse.Namespace) -> None:
    now = stamp()
    with immediate(conn):
        row = conn.execute("SELECT id FROM experiment_gaps WHERE id = ?", (args.id,)).fetchone()
        if row is None:
            raise SystemExit(f"unknown experiment gap id: {args.id}")
        conn.execute(
            """
            UPDATE experiment_gaps
            SET status = ?, reviewer_note = ?, updated_at = ?
            WHERE id = ?
            """,
            (args.status, args.note, now, args.id),
        )
    emit(conn.execute("SELECT * FROM experiment_gaps WHERE id = ?", (args.id,)).fetchone())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.set_defaults(func=cmd_init)

    add = sub.add_parser("add")
    add.add_argument("--review-pass", default="v3-sqlite-pass")
    add.add_argument("--source-file", default=DEFAULT_SOURCE)
    add.add_argument("--section", required=True)
    add.add_argument("--anchor", required=True)
    add.add_argument("--comment", required=True)
    add.add_argument("--instruction", required=True)
    add.add_argument("--priority", type=int, choices=range(4), default=1)
    add.add_argument("--actor", default="pixel")
    add.set_defaults(func=cmd_add)

    claim = sub.add_parser("claim")
    claim.add_argument("--worker", required=True)
    claim.add_argument("--lease-minutes", type=int, default=30)
    claim.set_defaults(func=cmd_claim)

    implement = sub.add_parser("implement")
    implement.add_argument("id", type=int)
    implement.add_argument("--worker", required=True)
    implement.add_argument("--summary", required=True)
    implement.add_argument("--evidence", required=True)
    implement.set_defaults(func=cmd_implement)

    verify = sub.add_parser("verify")
    verify.add_argument("id", type=int)
    verify.add_argument("--reviewer", default="pixel")
    verify.add_argument("--note", required=True)
    verify.set_defaults(func=cmd_verify)

    reopen = sub.add_parser("reopen")
    reopen.add_argument("id", type=int)
    reopen.add_argument("--reviewer", default="pixel")
    reopen.add_argument("--reason", required=True)
    reopen.set_defaults(func=cmd_reopen)

    decline = sub.add_parser("decline")
    decline.add_argument("id", type=int)
    decline.add_argument("--reviewer", default="pixel")
    decline.add_argument("--reason", required=True)
    decline.set_defaults(func=cmd_decline)

    listing = sub.add_parser("list")
    listing.add_argument(
        "--status",
        choices=("all", "open", "claimed", "implemented", "verified", "reopened", "declined"),
        default="all",
    )
    listing.set_defaults(func=cmd_list)

    show = sub.add_parser("show")
    show.add_argument("id", type=int)
    show.set_defaults(func=cmd_show)

    check = sub.add_parser("check")
    check.add_argument("id", type=int)
    check.set_defaults(func=cmd_check)

    stats = sub.add_parser("stats")
    stats.set_defaults(func=cmd_stats)

    history = sub.add_parser("history")
    history.add_argument("id", type=int)
    history.set_defaults(func=cmd_history)

    export = sub.add_parser("export")
    export.add_argument("--output", type=Path)
    export.set_defaults(func=cmd_export)

    gap_add = sub.add_parser("gap-add")
    gap_add.add_argument("--title", required=True)
    gap_add.add_argument("--claim", required=True)
    gap_add.add_argument("--present", required=True)
    gap_add.add_argument("--missing", required=True)
    gap_add.add_argument("--minimum", required=True)
    gap_add.add_argument("--priority", type=int, choices=range(4), default=1)
    gap_add.add_argument("--actor", default="pixel")
    gap_add.set_defaults(func=cmd_gap_add)

    gap_list = sub.add_parser("gap-list")
    gap_list.add_argument(
        "--status",
        choices=("all", "candidate", "selected", "running", "completed", "deferred", "not_needed"),
        default="all",
    )
    gap_list.set_defaults(func=cmd_gap_list)

    gap_update = sub.add_parser("gap-update")
    gap_update.add_argument("id", type=int)
    gap_update.add_argument(
        "--status",
        required=True,
        choices=("candidate", "selected", "running", "completed", "deferred", "not_needed"),
    )
    gap_update.add_argument("--note", required=True)
    gap_update.set_defaults(func=cmd_gap_update)
    return p


def main() -> None:
    args = parser().parse_args()
    conn = connect(args.db.resolve())
    try:
        args.func(conn, args)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
