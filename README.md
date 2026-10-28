# Boogle

## Local setup

Create and initialize the PostgreSQL database:

```sh
createdb boogle
psql -d boogle -f schema_handout.sql
```

Install dependencies and run the Flask app:

```sh
python -m pip install -r requirements.txt
python app.py
```

The app defaults to the local `boogle` database. To use another PostgreSQL
server, export a connection URL before starting it:

```sh
export DATABASE_URL='postgresql://user:password@localhost:5432/boogle'
```
