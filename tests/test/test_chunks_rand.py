import json
import random
from pathlib import Path

# Load json
json_path = Path("data/exports/chunks_raw_structured.json")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect all children (to be able to print parent)
parents_map = {}
children = []

for item in data.get("chunks", []):
    if item["chunk_level"] == "parent":
        parents_map[item["chunk_id"]] = item
        for child in item.get("children", []):
            children.append((child, item))

if not children:
    print("Không tìm thấy children chunks!")
    exit(0)

# Random pick 3 children
picks = random.sample(children, min(3, len(children)))

for i, (child, parent) in enumerate(picks, 1):
    print(f"\n{'='*20} CHUNK {i} {'='*20}")
    print(">>> NỘI DUNG CHUNK (CONTENT):")
    print(child["content"])
    print("\n>>> METADATA:")
    for key in ["chunk_id", "source", "section_name", "program_name", "ma_nganh"]:
        print(f" - {key:12}: {child.get(key)}")
    print("\n>>> TỪ PARENT CÓ NỘI DUNG:")
    # Print max 300 chars of parent to save space
    parent_cut = parent['content'][:300] + ("..." if len(parent['content']) > 300 else "")
    print(parent_cut)
    print("="*50)
