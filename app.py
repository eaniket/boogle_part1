
import os
import uuid
from datetime import timezone

import psycopg
from flask import Flask, jsonify, make_response, redirect, render_template, request

app = Flask(__name__)

COOKIE_NAME = "boogle_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days


def get_database_connection():
    """Open a PostgreSQL connection."""
    return psycopg.connect(
        os.environ.get("DATABASE_URL", "postgresql:///boogle")
    )


def format_timestamp(ts):
    """Format a timestamp as UTC ISO 8601 with milliseconds."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def get_client_id():
    """Reuse a valid browser UUID or generate a new one."""
    cookie_value = request.cookies.get(COOKIE_NAME)

    try:
        if cookie_value:
            return uuid.UUID(cookie_value)
    except (ValueError, TypeError, AttributeError):
        pass

    return uuid.uuid4()


def save_client(client_id):
    """Ensure the client exists in PostgreSQL."""
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO clients (client_id)
                VALUES (%s)
                ON CONFLICT (client_id)
                DO UPDATE SET last_seen = now()
                """,
                (client_id,),
            )


def set_client_cookie(response, client_id):
    """Set the persistent browser identity cookie."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=str(client_id),
        max_age=COOKIE_MAX_AGE,
        path="/",
        samesite="Lax",
        secure=False,
    )
    return response


@app.get("/")
def index():
    client_id = get_client_id()
    save_client(client_id)

    response = make_response(render_template("boogle.html"))
    return set_client_cookie(response, client_id)


@app.get("/search")
def search():
    # Do not strip or otherwise modify the user's query.
    query = request.args.get("q")

    if query is None or query == "":
        return redirect("/", code=303)

    client_id = get_client_id()

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            # Ensure the cookie's UUID exists in the clients table.
            cursor.execute(
                """
                INSERT INTO clients (client_id)
                VALUES (%s)
                ON CONFLICT (client_id)
                DO UPDATE SET last_seen = now()
                """,
                (client_id,),
            )

            cursor.execute(
                """
                INSERT INTO requests (client_id, ip, referer, path)
                VALUES (%s, %s, %s, %s)
                RETURNING request_id
                """,
                (
                    client_id,
                    request.remote_addr,
                    request.referrer,
                    request.path,
                ),
            )
            request_id = cursor.fetchone()[0]

            cursor.execute(
                """
                INSERT INTO searches (client_id, request_id, query)
                VALUES (%s, %s, %s)
                """,
                (client_id, request_id, query),
            )

    response = redirect("/", code=303)
    return set_client_cookie(response, client_id)


@app.get("/api/history")
def api_history():
    cookie_value = request.cookies.get(COOKIE_NAME)

    if not cookie_value:
        return jsonify({"client_id": None, "searches": []})

    try:
        client_id = uuid.UUID(cookie_value)
    except (ValueError, TypeError, AttributeError):
        return jsonify({"client_id": None, "searches": []})

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT query, ts
                FROM searches
                WHERE client_id = %s
                ORDER BY ts DESC, search_id DESC
                """,
                (client_id,),
            )
            rows = cursor.fetchall()

    searches = [
        {
            "query": query,
            "ts": format_timestamp(ts),
        }
        for query, ts in rows
    ]

    return jsonify(
        {
            "client_id": str(client_id),
            "searches": searches,
        }
    )


@app.get("/dump")
def dump():
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            # Include every client, even if it has no searches.
            cursor.execute(
                """
                SELECT client_id, first_seen, last_seen
                FROM clients
                ORDER BY client_id
                """
            )
            client_rows = cursor.fetchall()

            # Newest requests first within each client.
            cursor.execute(
                """
                SELECT client_id, ip, ts, request_id
                FROM requests
                ORDER BY client_id, ts DESC, request_id DESC
                """
            )
            request_rows = cursor.fetchall()

            # Newest searches first within each client.
            cursor.execute(
                """
                SELECT client_id, query, ts, search_id
                FROM searches
                ORDER BY client_id, ts DESC, search_id DESC
                """
            )
            search_rows = cursor.fetchall()

    requests_by_client = {}

    for client_id, ip, ts, request_id in request_rows:
        entry = requests_by_client.setdefault(
            client_id,
            {"latest_ip": None, "ips": []},
        )

        if ip is not None:
            ip_string = str(ip)

            if entry["latest_ip"] is None:
                entry["latest_ip"] = ip_string

            if ip_string not in entry["ips"]:
                entry["ips"].append(ip_string)

    searches_by_client = {}

    for client_id, query, ts, search_id in search_rows:
        searches_by_client.setdefault(client_id, []).append(
            {
                # Preserve the query exactly as stored.
                "query": query,
                "ts": format_timestamp(ts),
            }
        )

    clients = []

    for client_id, first_seen, last_seen in client_rows:
        request_info = requests_by_client.get(
            client_id,
            {"latest_ip": None, "ips": []},
        )

        clients.append(
            {
                "client_id": str(client_id),
                "first_seen": format_timestamp(first_seen),
                "last_seen": format_timestamp(last_seen),
                "latest_ip": request_info["latest_ip"],
                "ips": request_info["ips"],
                "searches": searches_by_client.get(client_id, []),
            }
        )

    return jsonify({"clients": clients})


@app.get("/jsontest")
def jsontest():
    return {"data": "I am in CSE190/CSE291!"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)