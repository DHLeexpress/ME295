#!/bin/bash
# usage: ghcode.sh outfile "query1" "query2" ...  (read-only GitHub code search via gh; sleeps to respect 10/min)
out=$1; shift
for q in "$@"; do
  echo "### $q" >> "$out"
  gh api -X GET search/code -f q="$q" -f per_page=100 --jq '"total=\(.total_count)", (.items[] | "\(.repository.full_name)\t\(.path)\t\(.html_url)")' >> "$out" 2>&1
  sleep 7
done
