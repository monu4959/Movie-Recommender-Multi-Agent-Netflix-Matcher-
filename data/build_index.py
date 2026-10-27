"""
One-shot loader: CSV -> cleaned rows -> embeddings -> Postgres.
Run once after `docker compose up -d`.
    uv run python data/build_index.py
"""

import asyncio
import os
import sys
from pathlib import Path

import asyncpg
import pandas as pd
from dateutil import parser as dateparser
from openai import OpenAI
from pgvector.asyncpg import register_vector
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import settings  # noqa: E402

CSV_PATH = Path(__file__).parent / "netflix_data.csv"
BATCH_SIZE = 100  # embeddings per API call
CAST_CAP = 5     # top N cast members to include in combined_text


# ---------- Cleaning ----------

def split_comma_list(value) -> list[str]:
    """Split 'Foo, Bar, Baz' into ['Foo', 'Bar', 'Baz']. Handle NaN."""
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_duration(value) -> tuple[int | None, str | None]:
    """'90 min' -> (90, 'min'). '2 Seasons' -> (2, 'Seasons')."""
    if pd.isna(value) or not str(value).strip():
        return None, None
    parts = str(value).strip().split(" ", 1)
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0]), parts[1]
    except ValueError:
        return None, None


def parse_date(value):
    if pd.isna(value) or not str(value).strip():
        return None
    try:
        return dateparser.parse(str(value)).date()
    except (ValueError, TypeError):
        return None


def empty_to_none(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def build_combined_text(row: dict) -> str:
    """Structured labeled text for embedding — matches our design decision."""
    cast_sample = row["cast_members"][:CAST_CAP]
    parts = [
        f"Title: {row['title']}",
        f"Type: {row['type']}",
    ]
    if row["genres"]:
        parts.append(f"Genres: {', '.join(row['genres'])}")
    if row["director"]:
        parts.append(f"Director: {', '.join(row['director'])}")
    if cast_sample:
        parts.append(f"Cast: {', '.join(cast_sample)}")
    if row["description"]:
        parts.append(f"Description: {row['description']}")
    return "\n".join(parts)


def clean_dataframe(df: pd.DataFrame) -> list[dict]:
    """Normalize the CSV into insert-ready dicts."""
    rows = []
    for _, raw in df.iterrows():
        duration_val, duration_unit = parse_duration(raw.get("duration"))
        row = {
            "show_id": str(raw["show_id"]),
            "type": empty_to_none(raw.get("type")),
            "title": empty_to_none(raw.get("title")) or "Untitled",
            "director": split_comma_list(raw.get("director")),
            "cast_members": split_comma_list(raw.get("cast")),
            "country": split_comma_list(raw.get("country")),
            "date_added": parse_date(raw.get("date_added")),
            "release_year": int(raw["release_year"]) if pd.notna(raw.get("release_year")) else None,
            "rating": empty_to_none(raw.get("rating")),
            "duration_value": duration_val,
            "duration_unit": duration_unit,
            "genres": split_comma_list(raw.get("listed_in")),
            "description": empty_to_none(raw.get("description")),
        }
        row["combined_text"] = build_combined_text(row)
        rows.append(row)
    return rows


# ---------- Embedding ----------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(
        model=settings.openai_embed_model,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def embed_all(rows: list[dict]) -> None:
    client = OpenAI(api_key=settings.openai_api_key)
    total = len(rows)
    for start in range(0, total, BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [r["combined_text"] for r in batch]
        vectors = embed_batch(client, texts)
        for row, vec in zip(batch, vectors):
            row["embedding"] = vec
        print(f"  Embedded {min(start + BATCH_SIZE, total)}/{total}")


# ---------- DB insert ----------

INSERT_SQL = """
INSERT INTO movies (
    show_id, type, title, director, cast_members, country,
    date_added, release_year, rating, duration_value, duration_unit,
    genres, description, combined_text, embedding
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
) ON CONFLICT (show_id) DO UPDATE SET
    embedding = EXCLUDED.embedding,
    combined_text = EXCLUDED.combined_text
"""


async def insert_rows(rows: list[dict]) -> None:
    conn = await asyncpg.connect(settings.postgres_dsn)
    await register_vector(conn)
    try:
        await conn.execute("TRUNCATE movies")  # fresh build
        # Use executemany for bulk insert
        records = [
            (
                r["show_id"], r["type"], r["title"], r["director"],
                r["cast_members"], r["country"], r["date_added"],
                r["release_year"], r["rating"], r["duration_value"],
                r["duration_unit"], r["genres"], r["description"],
                r["combined_text"], r["embedding"],
            )
            for r in rows
        ]
        await conn.executemany(INSERT_SQL, records)
        print(f"  Inserted {len(rows)} rows")
    finally:
        await conn.close()


# ---------- Main ----------

async def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Copy netflix_data.csv into data/.")
        sys.exit(1)

    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows")

    print("Cleaning rows...")
    rows = clean_dataframe(df)

    print(f"Embedding {len(rows)} rows with {settings.openai_embed_model}...")
    embed_all(rows)

    print("Inserting into Postgres...")
    await insert_rows(rows)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())