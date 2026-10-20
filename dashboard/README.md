# Bot status dashboard

Deploy this folder as a separate Python service. Install `requirements.txt` from
this folder, provide `DATABASE_URL` pointing to the worker's status database, and
use this start command with `dashboard/` as the working directory:

```sh
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The worker creates the status tables when it first publishes an update. Until
then, or whenever the database is unavailable, the page displays “No data from
the bot yet”. The page refreshes every 30 seconds. `/api/status` provides the
same view as JSON, and `/healthz` checks the web service independently of the bot.

Database reads use read-only transactions. The service has no trading or
configuration endpoints and needs no broker credentials.
