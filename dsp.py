"""A DSP you can run locally and test when working on your SSP! 

    pip install flask
    python dsp.py           # http://localhost:9000 (or: python dsp.py 9001)

The rtb URL when running your SSP should be http://localhost:9000. This
file implements two routes:

    POST /<dsp>/bid          any name works for <dsp>. The DSP places one bid at 2.00 on
                             imp[0], and 204 (no bid) if your bidfloor is above
                             it.
    GET  /bill/<bid>?price=  receives the billing notice and returns 204.

This DSP prints every request and billing notice it receives. All bids are
ties, so your winner logic determines who wins.
"""

import json
import sys
import uuid

from flask import Flask, request

PRICE = 2.00                            # what every bid offers, CPM in USD
BILLED = {}                             # bid id -> times billed
CRID = "cr-%s" % uuid.uuid4().hex[:10]
AD = ('<div data-boogle-ad data-dsp="testing" data-crid="%s" data-bid="%s" style="'
      'box-sizing:border-box;width:300px;height:250px;padding:24px;color:#fff;'
      'background:#5f6368;font:600 22px system-ui,sans-serif">Test ad</div>')

app = Flask(__name__)
app.json.sort_keys = False              # keep keys in OpenRTB's order


@app.post("/<dsp>/bid", strict_slashes=False)
def bid(dsp):
    """One bid request: a bid, 204 (no bid), or 400 if we cannot bid at all."""
    try:
        bid_request = json.loads(request.get_data())
        imp = bid_request["imp"][0]
        if not (isinstance(imp["id"], str) and imp["id"].strip()):
            raise ValueError
    except Exception:
        why = "Not a validly formatted bid request: we need a JSON object whose imp[0].id is a non-empty string."
        print("[%s] refused: %s" % (dsp, why))
        return {"error": [why]}, 400

    print("[%s] bid request:\n%s" % (dsp, json.dumps(bid_request, indent=2)))
    floor = imp.get("bidfloor", 0)
    if isinstance(floor, (int, float)) and PRICE < floor:
        print("[%s] -> no bid: your floor %s is above %.2f" % (dsp, floor, PRICE))
        return "", 204

    bid_id = "b-%s" % uuid.uuid4().hex[:16]
    BILLED[bid_id] = 0
    print("[%s] -> bid %s at %.2f" % (dsp, bid_id, PRICE))
    return {"id": str(bid_request.get("id") or ""), "cur": "USD", "seatbid": [{
        "seat": "testing",
        "bid": [{
            "id": bid_id,
            "impid": imp["id"],
            "price": PRICE,
            "adm": AD % (CRID, bid_id),
            "adomain": ["testing.example"],
            "crid": CRID,
            "attr": [],
            "w": 300,
            "h": 250,
            "mtype": 1,
            "burl": "%sbill/%s?price=${AUCTION_PRICE}" % (request.host_url, bid_id),
        }],
    }]}


@app.get("/bill/<path:bid_id>")
def bill(bid_id):
    """One billing notice: always 204, with a warning printed if it looks wrong."""
    price = request.args.get("price", "")
    note = ""
    if bid_id not in BILLED:
        note = "  WARNING: this DSP never made that bid"
    elif BILLED[bid_id]:
        note = "  WARNING: already billed; bill a winner exactly once"
    if "AUCTION_PRICE" in price:
        note += "  WARNING: replace ${AUCTION_PRICE} with the price you charged"
    if bid_id in BILLED:
        BILLED[bid_id] += 1
    print("[bill] %s charged %s%s" % (bid_id, price or "(no price)", note))
    return "", 204


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    sys.stdout.reconfigure(line_buffering=True)     # print each line as it happens
    print("Testing DSP at http://localhost:%d -- POST /<dsp>/bid, bidding %.2f" % (port, PRICE))
    app.run("localhost", port)
