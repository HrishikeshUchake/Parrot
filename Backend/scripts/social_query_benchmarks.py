from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
USER = "albert336"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    acts = read_jsonl(BASE / "activities.jsonl")
    feed = read_jsonl(BASE / "feed.jsonl")
    msgs = read_jsonl(BASE / "messages.jsonl")
    posts = acts + feed

    rel = [
        r
        for r in posts
        if str(r.get("author_name", "")) == USER
        or USER in (r.get("liker_names") or [])
        or any(
            str(c.get("commenter_name", "")) == USER
            for c in (r.get("comments") or [])
            if isinstance(c, dict)
        )
    ]

    tag = Counter()
    eng = Counter()
    msg_partner = Counter()
    msg_topic = Counter()

    for r in rel:
        t = r.get("tag_list")
        arr = t if isinstance(t, list) else (
            str(t).split(",") if isinstance(t, str) else [])
        for x in arr:
            x = str(x).strip().lower()
            if x:
                tag[x] += 1
        for c in (r.get("comments") or []):
            if isinstance(c, dict):
                n = str(c.get("commenter_name", "")).strip()
                if n and n != USER:
                    eng[n] += 1

    keywords = [
        "work",
        "music",
        "book",
        "fitness",
        "health",
        "weekend",
        "food",
        "tech",
        "citylife",
        "mindset",
        "learning",
    ]

    for m in msgs:
        s = str(m.get("sender_name", ""))
        r = str(m.get("receiver_name", ""))
        text = str(m.get("text", "")).replace("TEXT:\n", "").lower()
        if USER not in (s, r):
            continue
        partner = r if s == USER else s
        if partner:
            msg_partner[partner] += 1
        for k in keywords:
            if k in text:
                msg_topic[k] += 1

    print(
        {
            "relevant_posts": len(rel),
            "top_tags": tag.most_common(8),
            "top_commenters_on_relevant_posts": eng.most_common(8),
            "top_message_partners": msg_partner.most_common(8),
            "message_topic_hits": msg_topic.most_common(8),
        }
    )


if __name__ == "__main__":
    main()
