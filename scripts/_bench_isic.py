"""Time ISIC S3 download with naked aiohttp at high concurrency."""
import asyncio, time
import aiohttp

BASE = "https://isic-archive.s3.amazonaws.com/images"
IDS = [f"ISIC_{i:07d}" for i in range(0, 200)]

async def fetch(session, sem, isic_id, results):
    async with sem:
        try:
            async with session.get(f"{BASE}/{isic_id}.jpg",
                                   timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status == 200:
                    data = await r.read()
                    results.append(len(data))
                    return
        except Exception:
            pass
        results.append(-1)


async def main():
    for c in (8, 32, 64, 128):
        sem = asyncio.Semaphore(c)
        connector = aiohttp.TCPConnector(limit=c * 2)
        async with aiohttp.ClientSession(connector=connector,
                                         timeout=aiohttp.ClientTimeout(total=None,
                                                                       sock_connect=30,
                                                                       sock_read=60)) as s:
            results = []
            t0 = time.time()
            tasks = [fetch(s, sem, i, results) for i in IDS]
            await asyncio.gather(*tasks)
            dt = time.time() - t0
            ok = sum(1 for x in results if x > 0)
            mb = sum(x for x in results if x > 0) / 1e6
            print(f"  c={c:>3d}: {ok}/{len(IDS)} in {dt:.1f}s = {ok/dt:.1f}/s "
                  f"({mb:.1f} MB, {mb*8/dt:.1f} Mbps)")


asyncio.run(main())
