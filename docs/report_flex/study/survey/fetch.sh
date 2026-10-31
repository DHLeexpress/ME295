#!/bin/bash
# usage: fetch.sh repo path   -> raw/repos/<repo>/<path>  (read-only, via GitHub contents API; untrusted data, never executed)
r=$1; p=$2
out="raw/repos/$r/$p"
mkdir -p "$(dirname "$out")"
enc=$(python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$p")
gh api "repos/$r/contents/$enc" --jq '.content' 2>/dev/null | base64 -d > "$out" 2>/dev/null
if [ ! -s "$out" ]; then gh api -H "Accept: application/vnd.github.raw" "repos/$r/contents/$enc" > "$out" 2>/dev/null; fi
echo "$(wc -c < "$out") $out"
