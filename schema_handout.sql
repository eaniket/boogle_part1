-- Boogle table schemas
-- TODO: Fill in schema and entries as needed! 

CREATE TABLE IF NOT EXISTS clients (
    client_id   UUID PRIMARY KEY,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS requests (
    request_id  BIGSERIAL PRIMARY KEY,
    client_id   UUID REFERENCES clients(client_id),
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    ip          INET,
    referer     TEXT,
    path        TEXT
);

CREATE TABLE IF NOT EXISTS searches (
    search_id   BIGSERIAL PRIMARY KEY,
    client_id   UUID REFERENCES clients(client_id),
    request_id  BIGINT REFERENCES requests(request_id),
    query       TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create indices for faster reverse order lookups 
CREATE INDEX IF NOT EXISTS searches_by_client ON searches (client_id, ts DESC, search_id DESC);
CREATE INDEX IF NOT EXISTS requests_by_client ON requests (client_id, ts DESC, request_id DESC);
