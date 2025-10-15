# scripts/export_temperature_json.py
import argparse, torch, json

ap = argparse.ArgumentParser()
ap.add_argument("--in_pt", required=True, help="temperature .pt saved by trainer (e.g., temp_fold0.pt)")
ap.add_argument("--out_json", required=True, help="where to write temperature.json")
args = ap.parse_args()

obj = torch.load(args.in_pt, map_location="cpu")
T = float(obj.get("T", 1.0)) if isinstance(obj, dict) else 1.0
with open(args.out_json, "w", encoding="utf-8") as f:
    json.dump({"temperature": T}, f, indent=2)
print(f"Wrote {args.out_json} with temperature={T:.4f}")
