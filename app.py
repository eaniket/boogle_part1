from flask import Flask, render_template
app = Flask(__name__)

@app.get('/')
def index():
    return render_template('boogle.html')

@app.get('/jsontest')
def jsontest():
    return {"data" : "I am in CSE190/CSE291!"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8000")
