"""Verify direct S3 image URLs work for ISIC."""
import requests
for isic_id in ["ISIC_0000000", "ISIC_0024306", "ISIC_0050000", "ISIC_2956586", "ISIC_9880108"]:
    url = f"https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg"
    r = requests.head(url, timeout=20)
    print(f"  {isic_id}: status={r.status_code} ct={r.headers.get('content-type')} "
          f"size={r.headers.get('content-length')}")
