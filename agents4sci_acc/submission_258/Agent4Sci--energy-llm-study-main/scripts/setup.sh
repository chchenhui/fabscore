#!/usr/bin/env bash
mkdir -p outputs/sanity outputs/logs
cp .env.example .env 2>/dev/null || echo ".env already exists"
echo "[setup] folders and env ready"
