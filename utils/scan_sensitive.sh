#!/bin/bash
echo "🔍 Scanning repo for sensitive patterns ..."

grep -r -n --color=always -E \
"(API[_-]?KEY|SECRET|PASSWORD|TOKEN|Bearer |Authorization|ssh-rsa|BEGIN RSA PRIVATE KEY|BEGIN OPENSSH PRIVATE KEY)" \
./ --exclude-dir=.git --exclude=*.png --exclude=*.jpg --exclude=*.mp4 --exclude=*.jsonl --exclude=*.csv

echo "✅ Scan finished."
