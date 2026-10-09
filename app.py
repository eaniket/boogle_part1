import os
import uuid
from datetime import timezone
import psycopg
from flask import Flask, render_template, make_response, request, redirect, jsonify

app = Flask(__name__)


def get_database_connection():
    """Open a PostgreSQL connection using DATABASE_URL when it is provided."""
    return psycopg.connect(os.environ.get("DATABASE_URL", "postgresql:///boogle"))

def format_timestamp(ts):
    """Format a PostgreSQL timestamp as UTC ISO 8601 with milliseconds."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)

    return ts.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


@app.get('/')
def index():
    response = make_response(render_template('boogle.html'))
    cookie_key = 'boogle_id'
    if cookie_key not in request.cookies:
        response.set_cookie(
            key=cookie_key,
            value=str(uuid.uuid4()),
            max_age=3600*24*30,  # Set the cookie to expire in 30 days
            path='/',
            samesite='Lax',
            secure=False
        )
    return response

@app.get('/search')
def search():
    query = request.args.get('q', '').strip()

    if query:
        client_id = request.cookies.get('boogle_id')
        try:
            client_id = uuid.UUID(client_id) if client_id else uuid.uuid4()
        except ValueError:
            client_id = uuid.uuid4()

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
                cursor.execute(
                    """
                    INSERT INTO requests (client_id, ip, referer, path)
                    VALUES (%s, %s, %s, %s)
                    RETURNING request_id
                    """,
                    (client_id, request.remote_addr, request.referrer, request.path),
                )
                request_id = cursor.fetchone()[0]
                cursor.execute(
                    """
                    INSERT INTO searches (client_id, request_id, query)
                    VALUES (%s, %s, %s)
                    """,
                    (client_id, request_id, query),
                )

        response = redirect('/', code=303)
        response.set_cookie(
            key='boogle_id',
            value=str(client_id),
            max_age=3600*24*30,
            path='/',
            samesite='Lax',
            secure=False,
        )
        return response

    return redirect('/', code=303)


@app.get('/api/history')
def api_history():
    cookie_value = request.cookies.get('boogle_id')

    # No cookie means a new client with no history.
    if not cookie_value:
        return jsonify({
            "client_id": None,
            "searches": []
        })

    try:
        client_id = uuid.UUID(cookie_value)
    except (ValueError, AttributeError):
        return jsonify({
            "client_id": None,
            "searches": []
        })

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
            "query": row[0],
            "ts": row[1].isoformat(timespec='milliseconds').replace('+00:00', 'Z')
                if row[1].tzinfo is not None
                else row[1].isoformat(timespec='milliseconds') + 'Z'
        }
        for row in rows
    ]

    return jsonify({
        "client_id": str(client_id),
        "searches": searches
    })


@app.get('/dump')
def dump():
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            # Retrieve every client, including clients with no searches.
            cursor.execute("""
                SELECT client_id, first_seen, last_seen
                FROM clients
                ORDER BY client_id
            """)
            client_rows = cursor.fetchall()

            # Retrieve requests, newest first for each client.
            cursor.execute("""
                SELECT client_id, ip, ts, request_id
                FROM requests
                ORDER BY client_id, ts DESC, request_id DESC
            """)
            request_rows = cursor.fetchall()

            # Retrieve all searches, newest first for each client.
            cursor.execute("""
                SELECT client_id, query, ts, search_id
                FROM searches
                ORDER BY client_id, ts DESC, search_id DESC
            """)
            search_rows = cursor.fetchall()

    # Build request history for each client.
    requests_by_client = {}

    for client_id, ip, ts, request_id in request_rows:
        entry = requests_by_client.setdefault(
            client_id,
            {"latest_ip": None, "ips": []}
        )

        if ip is not None:
            ip_string = str(ip)

            # The first non-null IP encountered is the latest known IP.
            if entry["latest_ip"] is None:
                entry["latest_ip"] = ip_string

            # Preserve order from newest to oldest and remove duplicates.
            if ip_string not in entry["ips"]:
                entry["ips"].append(ip_string)

    # Build search history for each client.
    searches_by_client = {}

    for client_id, query, ts, search_id in search_rows:
        searches_by_client.setdefault(client_id, []).append({
            "query": query,
            "ts": format_timestamp(ts)
        })

    # Assemble the final response.
    clients = []

    for client_id, first_seen, last_seen in client_rows:
        request_info = requests_by_client.get(
            client_id,
            {"latest_ip": None, "ips": []}
        )

        clients.append({
            "client_id": str(client_id),
            "first_seen": format_timestamp(first_seen),
            "last_seen": format_timestamp(last_seen),
            "latest_ip": request_info["latest_ip"],
            "ips": request_info["ips"],
            "searches": searches_by_client.get(client_id, [])
        })

    return jsonify({"clients": clients})

@app.get('/jsontest')
def jsontest():
    return {"data" : "I am in CSE190/CSE291!"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
