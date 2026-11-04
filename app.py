import functools
import json
import math
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, redirect, render_template, request
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

app = Flask(__name__)
logger = app.logger

COOKIE_NAME = "boogle_id"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days
RTB_BASE_URL = os.environ.get(
    "RTB_BASE_URL", "https://webtracking.sysnet.ucsd.edu/pa1/rtb/"
).rstrip("/")
# RTB_BASE_URL = os.environ.get(
#     "RTB_BASE_URL", "http://localhost:9000"
# ).rstrip("/")
DSP_NAMES = ("steady", "niche", "shady", "glitch")
BID_FLOOR = 0.10
AUCTION_TIMEOUT_MS = 1000
BILL_TIMEOUT_SECONDS = 5.0
COMPLAINT_THRESHOLD = 10
IMPRESSION_ID = "1"
AD_WIDTH = 300
AD_HEIGHT = 250
ROOT = Path(__file__).resolve().parent
PART3_DIR = ROOT.parent / "starter_3_4" / "part3"
PART3_MODEL_PATH = ROOT / "part3_model.joblib"
SEGMENT_TARGETS = (
    "gender",
    "age",
    "education",
    "household_income",
    "industry",
    "employment_status",
    "employer_size",
    "home_ownership",
    "parental_status",
)
AUCTIONS = {}
COMPLAINT_COUNTS = {}
BLOCKED_DOMAINS = set()
KNOWN_DOMAINS = set()
AUCTION_LOCK = threading.Lock()


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

    response = make_response(render_template("new_boogle.html"))
    return set_client_cookie(response, client_id)


@app.get("/search")
def search():
    # Do not strip or otherwise modify the user's query.
    query = request.args.get("q")

    if query is None or query.strip() == "":
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


def get_active_searches(client_id):
    """Return the searches available to the Part 3 segmentation model."""
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
            return [
                {"query": query, "ts": format_timestamp(ts)}
                for query, ts in cursor.fetchall()
            ]


def user_document(row):
    """The Part 3 feature representation, unchanged from ``main.py``."""
    queries = [str(item.get("query", "")).lower() for item in row["searches"]]
    ua = " ".join(row.get("user_agents", [])).lower()
    langs = " ".join(row.get("accept_languages", [])).lower()
    metadata = []
    for token in re.findall(r"[a-z0-9_]+", ua):
        metadata.append("ua_" + token)
    for token in re.findall(r"[a-z]+", langs):
        metadata.append("lang_" + token)
    metadata += [
        f"query_count_{min(len(queries) // 5, 15)}",
        f"ip_count_{min(len(row.get('ips', [])), 5)}",
        f"ua_count_{min(len(row.get('user_agents', [])), 4)}",
    ]
    return " query_sep ".join(queries) + " metadata " + " ".join(metadata)


@functools.lru_cache(maxsize=1)
def load_part3_model():
    """Load Part 3's fitted model, training and persisting it only if needed."""
    try:
        artifact = joblib_load(PART3_MODEL_PATH)
        return artifact["vectorizer"], artifact["models"]
    except (OSError, EOFError, KeyError, TypeError, ValueError):
        # A missing or incomplete cache is rebuilt from the supplied Part 3
        # data.  The final artifact makes this expensive work a one-time cost.
        pass

    train = pd.read_json(PART3_DIR / "train.jsonl.gz", lines=True)
    dev = pd.read_json(PART3_DIR / "dev.jsonl.gz", lines=True)
    test = pd.read_json(PART3_DIR / "test.jsonl.gz", lines=True)
    y_train = pd.read_csv(PART3_DIR / "train_labels.csv", dtype=str)
    y_dev = pd.read_csv(PART3_DIR / "dev_labels.csv", dtype=str)

    all_users = pd.concat([train, dev, test], ignore_index=True)
    documents = all_users.apply(user_document, axis=1).tolist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=240_000,
        strip_accents="unicode",
        dtype=np.float32,
    )
    features = vectorizer.fit_transform(documents)
    train_features = features[: len(train)]
    dev_features = features[len(train): len(train) + len(dev)]
    labels = pd.concat([y_train, y_dev], ignore_index=True)
    labeled_features = features[: len(train) + len(dev)]

    models = {}
    best_params = {}
    for target in SEGMENT_TARGETS:
        best_score = -1.0
        best_settings = None
        for c in (0.35, 0.7, 1.2):
            for class_weight in (None, "balanced"):
                candidate = LinearSVC(
                    C=c, class_weight=class_weight, dual="auto", max_iter=5000
                )
                candidate.fit(train_features, y_train[target])
                score = f1_score(
                    y_dev[target], candidate.predict(dev_features), average="macro"
                )
                if score > best_score:
                    best_score = score
                    best_settings = (c, class_weight)

        c, class_weight = best_settings
        best_params[target] = {"C": c, "class_weight": class_weight}
        model = LinearSVC(
            C=c, class_weight=class_weight, dual="auto", max_iter=5000
        )
        model.fit(labeled_features, labels[target])
        models[target] = model

    temporary_model_path = PART3_MODEL_PATH.with_suffix(".tmp")
    joblib_dump(
        {"vectorizer": vectorizer, "models": models, "best_params": best_params},
        temporary_model_path,
    )
    os.replace(temporary_model_path, PART3_MODEL_PATH)
    return vectorizer, models


def infer_segment_ids(client_id, searches):
    """Infer this visitor's segments with the exact supplied Part 3 model."""
    del client_id
    segment_path = ROOT / "segments.json"

    try:
        with segment_path.open(encoding="utf-8") as segment_file:
            taxonomy = json.load(segment_file)["segments"]
        segment_by_value = {
            (item["attribute"], item["value"]): str(item["id"])
            for item in taxonomy
        }
        visitor_row = {
            "searches": searches,
            "ips": [request.remote_addr] if request.remote_addr else [],
            "user_agents": [request.user_agent.string]
            if request.user_agent.string
            else [],
            "accept_languages": [request.headers.get("Accept-Language", "")],
        }
        vectorizer, models = load_part3_model()
        visitor_features = vectorizer.transform([user_document(visitor_row)])
        prediction = {}
        for target, model in models.items():
            scores = np.asarray(model.decision_function(visitor_features)).ravel()
            if not scores.size:
                continue
            if len(model.classes_) == 2 and scores.size == 1:
                value = model.classes_[1 if scores[0] >= 0 else 0]
            else:
                order = np.argsort(scores)
                value = model.classes_[order[-1]]
            # Part 3 trains one classifier for each attribute and emits its
            # selected class.  Its raw LinearSVC margins are not calibrated
            # probabilities, so an arbitrary cutoff would hide otherwise
            # usable predictions (including Niche's top_10 audience).
            prediction[target] = value
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return []

    # The taxonomy lookup omits any model value that cannot be represented and
    # guarantees no more than one segment for each demographic attribute.
    return [
        segment_id
        for attribute, value in prediction.items()
        for segment_id in [segment_by_value.get((attribute, value))]
        if segment_id is not None
    ]


def build_bid_request(client_id, segment_ids, auction_id=None, badv=None):
    auction_id = auction_id or uuid.uuid4().hex
    return {
        "id": auction_id,
        "imp": [
            {
                "id": IMPRESSION_ID,
                "tagid": "boogle-top",
                "banner": {"format": [{"w": AD_WIDTH, "h": AD_HEIGHT}]},
                "bidfloor": BID_FLOOR,
            }
        ],
        "device": {
            "ua": request.user_agent.string,
            "ip": request.remote_addr or "",
        },
        "user": {
            "id": str(client_id),
            "data": [
                {
                    "id": "boogle",
                    "name": "boogle-inferred",
                    "segment": [{"id": segment_id} for segment_id in segment_ids],
                    "ext": {"segtax": 500},
                }
            ],
        },
        "tmax": AUCTION_TIMEOUT_MS,
        "badv": badv or [],
    }


def request_dsp_bid(dsp_name, bid_request, deadline):
    """Fetch one DSP response and record when its complete body arrived."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return None

    http_request = Request(
        f"{RTB_BASE_URL}/{dsp_name}/bid",
        data=json.dumps(bid_request).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(http_request, timeout=remaining) as response:
            status = response.status
            body = response.read()
        arrived_at = time.monotonic()
    except HTTPError as exc:
        # urllib represents some non-200 responses as exceptions.  A 204 is a
        # normal no-bid; every other HTTP failure is ignored independently.
        logger.info(
            "Auction %s DSP response failed: dsp=%s http_status=%s",
            bid_request["id"],
            dsp_name,
            exc.code,
        )
        if exc.code == 204:
            return None
        return None
    except (URLError, TimeoutError, OSError, ValueError):
        logger.info(
            "Auction %s DSP response failed: dsp=%s reason=network_or_timeout",
            bid_request["id"],
            dsp_name,
        )
        return None

    if arrived_at > deadline or status == 204 or status != 200:
        logger.info(
            "Auction %s discarded DSP response: dsp=%s status=%s "
            "arrived_after_deadline=%s",
            bid_request["id"],
            dsp_name,
            status,
            arrived_at > deadline,
        )
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        logger.warning(
            "Auction %s discarded invalid JSON from dsp=%s",
            bid_request["id"],
            dsp_name,
        )
        return None
    if not isinstance(payload, dict) or payload.get("id") != bid_request["id"]:
        logger.warning(
            "Auction %s discarded mismatched response from dsp=%s response_id=%s",
            bid_request["id"],
            dsp_name,
            payload.get("id") if isinstance(payload, dict) else None,
        )
        return None
    payload["_dsp"] = dsp_name
    logger.info(
        "Auction %s accepted response from dsp=%s status=%s body_bytes=%d",
        bid_request["id"],
        dsp_name,
        status,
        len(body),
    )
    return payload


def request_all_bids(bid_request, deadline):
    executor = ThreadPoolExecutor(max_workers=len(DSP_NAMES))
    futures = [
        executor.submit(request_dsp_bid, dsp_name, bid_request, deadline)
        for dsp_name in DSP_NAMES
    ]
    # Keep waiting for workers to finish, but request_dsp_bid applies the
    # OpenRTB deadline and discards any response that arrives after tmax.
    # The auction is bounded by the OpenRTB tmax deadline. Each worker also
    # uses the remaining deadline as its socket timeout.
    done, pending = wait(
        futures,
        timeout=max(0, deadline - time.monotonic()),
    )
    for future in pending:
        future.cancel()
    # Do not wait for a timed-out network call while leaving the request.  Each
    # worker still has its own deadline-derived socket timeout.
    executor.shutdown(wait=False, cancel_futures=True)

    responses = []
    logger.info(
        "Auction %s DSP collection complete: done=%d pending=%d",
        bid_request["id"],
        len(done),
        len(pending),
    )
    for future in done:
        try:
            response = future.result()
        except Exception:
            print(f"Future raised an exception: {future.exception()}")
            response = None
        if response is not None:
            responses.append(response)
    return responses


def extract_valid_bids(responses):
    valid_bids = []
    for response in responses:
        if response.get("cur") != "USD":
            logger.info(
                "Rejected DSP response: dsp=%s reason=currency currency=%s",
                response.get("_dsp"),
                response.get("cur"),
            )
            continue
        seatbids = response.get("seatbid")
        if not isinstance(seatbids, list):
            logger.info(
                "Rejected DSP response: dsp=%s reason=seatbid_type",
                response.get("_dsp"),
            )
            continue

        for seatbid in seatbids:
            if not isinstance(seatbid, dict) or not isinstance(seatbid.get("bid"), list):
                continue
            for bid in seatbid["bid"]:
                if not isinstance(bid, dict):
                    continue
                price = bid.get("price")
                if (
                    isinstance(price, bool)
                    or not isinstance(price, (int, float))
                    or not math.isfinite(price)
                    or price < BID_FLOOR
                    or bid.get("impid") != IMPRESSION_ID
                    or bid.get("w") != AD_WIDTH
                    or bid.get("h") != AD_HEIGHT
                    or not isinstance(bid.get("adm"), str)
                    or not bid.get("adm")
                    or not isinstance(bid.get("burl"), str)
                    or not bid.get("burl")
                ):
                    logger.info(
                        "Rejected bid: dsp=%s price=%s impid=%s size=%sx%s "
                        "has_adm=%s has_burl=%s",
                        response.get("_dsp"),
                        price,
                        bid.get("impid"),
                        bid.get("w"),
                        bid.get("h"),
                        bool(bid.get("adm")),
                        bool(bid.get("burl")),
                    )
                    continue
                bid_copy = dict(bid)
                bid_copy["_dsp"] = response.get("_dsp")
                valid_bids.append(bid_copy)
                logger.info(
                    "Accepted bid: dsp=%s price=%.4f domain=%s crid=%s",
                    response.get("_dsp"),
                    price,
                    (bid.get("adomain") or [None])[0],
                    bid.get("crid"),
                )
    return valid_bids


def choose_winner_and_price(valid_bids):
    if not valid_bids:
        return None, None
    ranked = sorted(
        valid_bids,
        key=lambda bid: (
            -bid["price"],
            bid.get("_dsp") or "",
            (bid.get("adomain") or [""])[0],
        ),
    )
    winner = ranked[0]
    reference_price = ranked[1]["price"] if len(ranked) > 1 else None
    final_price = (
        min(winner["price"], max(ranked[1]["price"], BID_FLOOR) + 0.01)
        if len(ranked) > 1
        else BID_FLOOR + 0.01
    )
    logger.info(
        "Auction pricing: candidates=%s winner_price=%.4f "
        "runner_up_price=%s final_price=%.4f",
        [
            {
                "dsp": bid.get("_dsp"),
                "price": bid["price"],
                "domain": (bid.get("adomain") or [None])[0],
            }
            for bid in ranked
        ],
        winner["price"],
        reference_price,
        final_price,
    )
    return winner, final_price


def bill_winner(winner, final_price):
    billing_url = winner["burl"].replace(
        "${AUCTION_PRICE}", format(final_price, ".12g")
    )
    try:
        logger.info(
            "Billing winner: dsp=%s bid_price=%.4f auction_price=%.4f",
            winner.get("_dsp"),
            winner["price"],
            final_price,
        )
        with urlopen(billing_url, timeout=BILL_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", 200)
        logger.info(
            "Billing response: dsp=%s auction_price=%.4f status=%s",
            winner.get("_dsp"),
            final_price,
            status,
        )
        return 200 <= status < 300
    except HTTPError as exc:
        return 200 <= exc.code < 300
    except (URLError, TimeoutError, OSError, ValueError):
        return False


@app.post("/ad/feedback")
def ad_feedback():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON object required"}), 400
    auction_id = payload.get("auction_id")
    if (
        not isinstance(auction_id, str)
        or not auction_id
        or payload.get("event") != "complaint"
    ):
        return jsonify({"error": "invalid complaint"}), 400

    with AUCTION_LOCK:
        auction = AUCTIONS.get(auction_id)
        if auction is None:
            return jsonify({"error": "unknown auction_id"}), 404
        if auction["complaint_recorded"]:
            return "", 204
        auction["complaint_recorded"] = True
        domain = auction.get("domain")
        if domain:
            COMPLAINT_COUNTS[domain] = COMPLAINT_COUNTS.get(domain, 0) + 1
            # Block the advertiser that actually crossed the threshold. Do not
            # replace it with the globally highest-count advertiser, and do not
            # clear previously blocked domains.
            if COMPLAINT_COUNTS[domain] >= COMPLAINT_THRESHOLD:
                BLOCKED_DOMAINS.add(domain)
    return "", 204


def complaint_badv():
    """Return one complaint-heavy domain when known alternatives exist.

    Blocking only one advertiser at a time is deliberately conservative: the
    DSP pool can be small, and blocking every complained-about domain at once
    can leave the page with no ad inventory.
    """
    with AUCTION_LOCK:
        eligible_to_block = [
            domain
            for domain in BLOCKED_DOMAINS
            if COMPLAINT_COUNTS.get(domain, COMPLAINT_THRESHOLD)
            >= COMPLAINT_THRESHOLD
        ]
        alternatives = KNOWN_DOMAINS - set(eligible_to_block)

        if not eligible_to_block or not alternatives:
            return []

        # Prioritize the advertiser with the most complaints. Stable sorting
        # makes the choice deterministic when counts are tied.
        worst_domain = max(
            sorted(eligible_to_block),
            key=lambda domain: COMPLAINT_COUNTS.get(domain, 0),
        )
        return [worst_domain]


@app.get("/ad")
def ad():
    cookie_value = request.cookies.get(COOKIE_NAME)
    try:
        client_id = uuid.UUID(cookie_value) if cookie_value else uuid.uuid4()
    except (ValueError, TypeError, AttributeError):
        client_id = uuid.uuid4()

    try:
        searches = get_active_searches(client_id)
    except (psycopg.Error, OSError):
        searches = []

    auction_id = uuid.uuid4().hex
    segment_ids = infer_segment_ids(client_id, searches)
    blocked_domains = complaint_badv()
    bid_request = build_bid_request(
        client_id, segment_ids, auction_id=auction_id, badv=blocked_domains
    )
    deadline = time.monotonic() + AUCTION_TIMEOUT_MS / 1000
    responses = request_all_bids(bid_request, deadline)
    print(f"Received responses from DSPs: {responses}")

    valid_bids = extract_valid_bids(responses)

    # Learn the advertiser pool from every valid bid, not only the winner.
    # This helps avoid blocking the only advertiser known to the SSP.
    with AUCTION_LOCK:
        for bid in valid_bids:
            bid_domains = bid.get("adomain") or []
            if isinstance(bid_domains, list):
                KNOWN_DOMAINS.update(
                    domain
                    for domain in bid_domains
                    if isinstance(domain, str) and domain
                )

    winner, final_price = choose_winner_and_price(valid_bids)

    if winner is None:
        return "", 200

    domains = winner.get("adomain") or []
    valid_domains = (
        [domain for domain in domains if isinstance(domain, str) and domain]
        if isinstance(domains, list)
        else []
    )
    with AUCTION_LOCK:
        KNOWN_DOMAINS.update(valid_domains)
        AUCTIONS[auction_id] = {
            "domain": valid_domains[0] if valid_domains else None,
            "complaint_recorded": False,
            "billed": False,
        }
    billed = bill_winner(winner, final_price)
    with AUCTION_LOCK:
        AUCTIONS[auction_id]["billed"] = billed
    if not billed:
        logger.warning(
            "Auction %s suppressed because billing failed: dsp=%s",
            auction_id,
            winner.get("_dsp"),
        )
        return "", 200
    return make_response(winner["adm"], 200, {"Content-Type": "text/html; charset=utf-8"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)