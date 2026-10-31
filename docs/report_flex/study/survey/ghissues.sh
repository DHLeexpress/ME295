#!/bin/bash
# usage: ghissues.sh outfile repo "query1" ... (read-only issue/PR search)
out=$1; repo=$2; shift 2
for q in "$@"; do
  echo "### $q" >> "$out"
  gh api -X GET search/issues -f q="repo:$repo $q" -f per_page=100 --jq '"total=\(.total_count)", (.items[] | "#\(.number)\t\(.state)\t\(.created_at[:10])\t\(if .pull_request then "PR" else "issue" end)\t\(.title)")' >> "$out" 2>&1
  sleep 2.5
done
