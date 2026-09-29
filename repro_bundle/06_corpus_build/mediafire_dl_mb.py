#!/usr/bin/env python3
"""Server-side MediaFire fetch (Motionbuilder-friendly CMU BVH = STANDARD CMU skeleton, matches cmu_all_perform).
Uses the ?<key> short URL (301-redirects to the real file page), decodes data-scrambled-url, derives the true
filename from the direct link, downloads, verifies ZIP magic."""
import urllib.request, re, base64, sys, os, ssl

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

# Motionbuilder-friendly 2010 re-release keys, by directory range
KEYS = ["bknr4qekg9nqyat", "t3dr1uey8xtnd2n", "fsgbtcwi7w1pxxx", "5cgy081zdfyvtkf",
        "4i6yai58lwy5je7", "r6y8sc60mabhneu", "6xqillndtwfw65d", "u4yfyyvq8yyok93", "88y4in1hspoi46e"]
OUT = "/hpcstor6/scratch01/p/p.bendiksen001/virtual_reality/triplets/datasets/_cmu_mb_raw"
os.makedirs(OUT, exist_ok=True)

def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)

def direct_link(key):
    html = get("https://www.mediafire.com/?" + key).read().decode("utf-8", "ignore")  # follows redirect
    m = re.search(r'data-scrambled-url="([^"]+)"', html)
    if m:
        try:
            dec = base64.b64decode(m.group(1)).decode("utf-8", "ignore")
            if dec.startswith("http") and ".zip" in dec:
                return dec
        except Exception:
            pass
    for pat in [r'href="(https://download[0-9]+\.mediafire\.com/[^"]+\.zip[^"]*)"',
                r'(https://download[0-9]+\.mediafire\.com/[^"\s]+\.zip)']:
        mm = re.findall(pat, html)
        if mm:
            return mm[0]
    return None

for key in KEYS:
    try:
        link = direct_link(key)
        if not link:
            print("NO-LINK", key); continue
        name = link.split("?")[0].split("/")[-1]
        dest = os.path.join(OUT, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 100000:
            print("SKIP (have)", name); continue
        data = get(link, timeout=240).read()
        if data[:2] != b"PK":
            print("NOT-ZIP", key, name, "first:", data[:16]); continue
        open(dest, "wb").write(data)
        print("OK", name, "%.1f MB" % (len(data) / 1e6))
    except Exception as e:
        print("ERR", key, str(e)[:120])
