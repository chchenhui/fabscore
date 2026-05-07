#!/usr/bin/env python3
"""Start a local PostgreSQL server with TCP on localhost:5432.

Uses pgserver's bundled PostgreSQL binaries (initdb, pg_ctl) directly since
Docker is unavailable. The pgserver.PostgresServer class disables TCP on Linux
(passes -h ""), so we drive initdb/pg_ctl ourselves to enable TCP listening.

When running as root, commands are executed as the 'pgserver' user (created
automatically if needed), since PostgreSQL refuses to run as root.

Usage:
    python start_postgres.py          # start server (foreground, blocks)
    python start_postgres.py --stop   # stop the running server
    python start_postgres.py --status # check server status
"""

import os
import pwd
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ["LANG"] = "C.utf8"
os.environ["LC_ALL"] = "C.utf8"

PGDATA = Path(__file__).parent / "pgdata"
PG_PORT = 5432
PG_USER = "postgres"
SYSTEM_USER = None

BIN_DIR = None
for site in [
    Path(__file__).parent.parent / ".venv" / "lib",
    Path(sys.prefix) / "lib",
]:
    candidates = list(site.glob("**/pgserver/pginstall/bin/pg_ctl"))
    if candidates:
        BIN_DIR = candidates[0].parent
        break

if BIN_DIR is None:
    sys.exit("ERROR: Could not find pgserver binaries. Is pgserver installed?")

INITDB = str(BIN_DIR / "initdb")
PG_CTL = str(BIN_DIR / "pg_ctl")


def ensure_system_user():
    global SYSTEM_USER
    if os.geteuid() != 0:
        return
    SYSTEM_USER = "pgserver"
    try:
        pwd.getpwnam(SYSTEM_USER)
    except KeyError:
        subprocess.run(
            ["useradd", "-s", "/bin/bash", SYSTEM_USER],
            check=True, capture_output=True, text=True,
        )


def run_as_user(cmd, **kwargs):
    if SYSTEM_USER is not None:
        pw = pwd.getpwnam(SYSTEM_USER)
        kwargs["user"] = pw.pw_uid
    return subprocess.run(cmd, **kwargs)


def is_port_open(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        s.close()
        return True
    except OSError:
        return False


def chown_recursive(path: Path):
    if SYSTEM_USER is None:
        return
    pw = pwd.getpwnam(SYSTEM_USER)
    uid, gid = pw.pw_uid, pw.pw_gid
    os.chown(path, uid, gid)
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chown(os.path.join(root, d), uid, gid)
        for f in files:
            os.chown(os.path.join(root, f), uid, gid)


def init_pgdata():
    if (PGDATA / "PG_VERSION").exists():
        print(f"pgdata already initialized at {PGDATA}")
        return
    PGDATA.mkdir(parents=True, exist_ok=True)
    chown_recursive(PGDATA)

    print(f"Running initdb at {PGDATA} ...")
    run_as_user(
        [INITDB, "-D", str(PGDATA), "--auth=trust", "--auth-local=trust",
         "--encoding=utf8", "-U", PG_USER],
        check=True,
    )

    conf = PGDATA / "postgresql.conf"
    text = conf.read_text()
    text = text.replace("#listen_addresses = 'localhost'", "listen_addresses = 'localhost'")
    text = text.replace(f"#port = {PG_PORT}", f"port = {PG_PORT}")
    conf.write_text(text)

    pg_hba = PGDATA / "pg_hba.conf"
    hba_text = pg_hba.read_text()
    if "host all all 127.0.0.1/32 trust" not in hba_text:
        hba_text += "\nhost all all 127.0.0.1/32 trust\n"
        hba_text += "host all all ::1/128 trust\n"
        pg_hba.write_text(hba_text)

    chown_recursive(PGDATA)
    print("pgdata initialized with TCP on localhost:5432")


def start_server():
    log_file = PGDATA / "server.log"
    run_as_user(
        [PG_CTL, "-D", str(PGDATA), "-l", str(log_file), "-w", "start"],
        check=True,
    )


def stop_server():
    run_as_user(
        [PG_CTL, "-D", str(PGDATA), "-m", "fast", "stop"],
        check=True,
    )


def server_status():
    result = run_as_user(
        [PG_CTL, "-D", str(PGDATA), "status"],
        capture_output=True, text=True,
    )
    print(result.stdout.strip())
    return result.returncode == 0


def create_marketplace_db():
    createdb = BIN_DIR / "createdb"
    result = run_as_user(
        [str(createdb), "-h", "localhost", "-p", str(PG_PORT), "-U", PG_USER, "marketplace"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Created database 'marketplace'")
    elif "already exists" in result.stderr:
        print("Database 'marketplace' already exists")
    else:
        print(f"createdb output: {result.stderr.strip()}")


def main():
    ensure_system_user()

    if "--stop" in sys.argv:
        print("Stopping PostgreSQL...")
        stop_server()
        return

    if "--status" in sys.argv:
        server_status()
        return

    init_pgdata()

    if is_port_open("localhost", PG_PORT):
        print(f"Port {PG_PORT} already in use. Checking if it's our server...")
        if server_status():
            print("Server already running.")
        else:
            sys.exit(f"ERROR: Port {PG_PORT} in use by another process")
    else:
        print("Starting PostgreSQL...")
        start_server()

    for i in range(10):
        if is_port_open("localhost", PG_PORT):
            print(f"PostgreSQL is listening on localhost:{PG_PORT}")
            break
        time.sleep(1)
    else:
        sys.exit(f"ERROR: PostgreSQL TCP port {PG_PORT} not reachable after 10s")

    create_marketplace_db()

    print("Server running. Press Ctrl+C to stop.")

    def handle_signal(sig, frame):
        print("\nShutting down PostgreSQL...")
        try:
            stop_server()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutting down PostgreSQL...")
        stop_server()


if __name__ == "__main__":
    main()
