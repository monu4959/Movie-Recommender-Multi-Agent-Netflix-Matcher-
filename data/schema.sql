-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Main movies table: denormalized with arrays, one row per title
CREATE TABLE IF NOT EXISTS movies (
    show_id         TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    title           TEXT NOT NULL,
    director        TEXT[],
    cast_members    TEXT[],
    country         TEXT[],
    date_added      DATE,
    release_year    INTEGER,
    rating          TEXT,
    duration_value  INTEGER,
    duration_unit   TEXT,
    genres          TEXT[],
    description     TEXT,

    -- Derived fields
    combined_text   TEXT,
    embedding       vector(1536),

    -- Full-text search column, populated by trigger (below)
    search_tsv      tsvector
);

-- Trigger function: build the weighted tsvector on insert/update
CREATE OR REPLACE FUNCTION movies_update_search_tsv() RETURNS trigger AS $$
BEGIN
    NEW.search_tsv :=
        setweight(to_tsvector('english', coalesce(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(array_to_string(NEW.cast_members, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(array_to_string(NEW.genres, ' '), '')), 'B') ||
        setweight(to_tsvector('english', coalesce(NEW.description, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER movies_tsv_trigger
    BEFORE INSERT OR UPDATE ON movies
    FOR EACH ROW
    EXECUTE FUNCTION movies_update_search_tsv();

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS movies_embedding_hnsw_idx
    ON movies USING hnsw (embedding vector_cosine_ops);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS movies_search_tsv_idx
    ON movies USING gin(search_tsv);

-- GIN indexes for array containment queries
CREATE INDEX IF NOT EXISTS movies_genres_idx
    ON movies USING gin(genres);

CREATE INDEX IF NOT EXISTS movies_cast_idx
    ON movies USING gin(cast_members);

CREATE INDEX IF NOT EXISTS movies_director_idx
    ON movies USING gin(director);

-- B-tree indexes for filter columns
CREATE INDEX IF NOT EXISTS movies_type_idx         ON movies (type);
CREATE INDEX IF NOT EXISTS movies_release_year_idx ON movies (release_year);