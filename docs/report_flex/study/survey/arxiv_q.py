import re, sys, urllib.parse, urllib.request, time, json, os
out = sys.argv[1]
qs = sys.argv[2:]
res = {}
for q in qs:
    url = 'https://export.arxiv.org/api/query?' + urllib.parse.urlencode({'search_query': q, 'start': 0, 'max_results': 100, 'sortBy': 'submittedDate', 'sortOrder': 'descending'})
    s = urllib.request.urlopen(url, timeout=90).read().decode()
    entries = s.split('<entry>')[1:]
    lst = []
    for e in entries:
        t = ' '.join(re.search(r'<title>(.*?)</title>', e, re.S).group(1).split())
        i = re.search(r'<id>(.*?)</id>', e).group(1)
        p = re.search(r'<published>(.*?)</published>', e).group(1)[:10]
        a = ' '.join(re.search(r'<summary>(.*?)</summary>', e, re.S).group(1).split())
        lst.append(dict(id=i, date=p, title=t, abstract=a))
    res[q] = lst
    print(f'## {q}: {len(lst)}')
    for x in lst:
        print(x['date'], x['id'], x['title'])
    time.sleep(3.5)
json.dump(res, open(out, 'w'), indent=1)
