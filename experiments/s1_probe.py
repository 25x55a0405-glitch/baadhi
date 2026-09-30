"""Step 1 probe: which Sentinel-1/2 scenes cover the Trishuli corridor around the 26 Aug 2026 event?"""
import sys
import pystac_client, planetary_computer as pc

BBOX = [85.05, 27.78, 85.50, 28.32]          # Rasuwagadhi (Bhote Koshi) down to Trishuli Bazar / Devighat
WHEN = sys.argv[1] if len(sys.argv) > 1 else "2026-07-20/2026-09-30"
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

for coll in ["sentinel-1-rtc", "sentinel-1-grd"]:
    items = sorted(cat.search(collections=[coll], bbox=BBOX, datetime=WHEN).items(), key=lambda i: i.datetime)
    print(f"\n== {coll}: {len(items)} scenes")
    for it in items:
        p = it.properties
        print(f"  {it.datetime:%Y-%m-%d %H:%M}  track {p.get('sat:relative_orbit')}  {p.get('sat:orbit_state')}  {p.get('platform')}  pol {p.get('sar:polarizations')}  {it.id[:60]}")

items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=BBOX, datetime=WHEN).items(), key=lambda i: i.datetime)
print(f"\n== sentinel-2-l2a: {len(items)} scenes (cloud %)")
for it in items:
    print(f"  {it.datetime:%Y-%m-%d}  cloud {it.properties.get('eo:cloud_cover'):5.1f}%  tile {it.properties.get('s2:mgrs_tile')}")

dem = list(cat.search(collections=["cop-dem-glo-30"], bbox=BBOX).items())
print(f"\n== cop-dem-glo-30: {len(dem)} tiles: {[d.id for d in dem]}")
