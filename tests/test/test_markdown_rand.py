import json
import random
from pathlib import Path

# Load json
json_path = Path("data/exports/chunks_raw_markdown.json")
if not json_path.exists():
    print(f"❌ Không tìm thấy file {json_path}. Bạn nhớ chạy export trước nhé:")
    print("   python ingestion/export_chunks.py --type markdown")
    exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Collect parents and children
parents_map = {}
children = []

for item in data.get("chunks", []):
    if item.get("chunk_level") == "parent":
        parents_map[item["chunk_id"]] = item
        # Lấy children từ array nested trong json
        for child in item.get("children", []):
            children.append((child, item))

if not children:
    print("❌ Không tìm thấy children chunks nào trong Markdown!")
    exit(0)

# Random pick 3 children
picks = random.sample(children, min(3, len(children)))

print("\n" + "🚀" * 30)
print("  TEST THỬ 3 CHUNK NGẪU NHIÊN TỪ MARKDOWN")
print("🚀" * 30)

for i, (child, parent) in enumerate(picks, 1):
    print(f"\n==================== MẪU {i} ====================")
    
    print("\n📦 NỘI DUNG CHUNK HIỆN TẠI (Được đưa đi Vector Search):")
    print("-" * 50)
    print(child.get("content", ""))
    print("-" * 50)
    
    print("\n⚙️  METADATA:")
    print(f"   • File gốc     : {child.get('source')}")
    print(f"   • Section Name : {child.get('section_name')}")
    # Markdown thì overlap_tokens > 0 do SemanticChunker cắt
    print(f"   • Overlap Token: {child.get('overlap_tokens', 0)} (Chồng lấn câu)")
    print(f"   • Token Count  : {child.get('token_count', 0)} tokens")
    
    print("\n👨‍🦳 PARENT CHUNK CHỨA NÓ (Full context):")
    parent_cut = parent.get('content', '')[:300] + ("..." if len(parent.get('content', '')) > 300 else "")
    print(f"   {parent_cut}")
    print("==================================================")
