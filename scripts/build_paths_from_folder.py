import argparse, os, re, csv
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)

pat = re.compile(r'(ISIC[_-]?\d{6,})', re.I)
rows = []
for p in Path(args.root).rglob('*.jpg'):
    m = pat.search(p.name)
    if m:
        isic_id = m.group(1).replace('-', '_').upper()
        if not isic_id.startswith('ISIC_'):
            isic_id = 'ISIC_' + re.sub(r'\D', '', isic_id)
        rows.append((isic_id, str(p)))
with open(args.out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['isic_id','filepath']); w.writerows(rows)
print('Wrote', args.out, 'rows=', len(rows))
