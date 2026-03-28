import json
import os
from pathlib import Path

# Cấu hình token
MAX_TOKEN_WARN = 800  # Cảnh báo nếu chunk > 800 token (BGE-m3 max 1024, an toàn nên để 800)
MIN_TOKEN_WARN = 10   # Cảnh báo nếu chunk < 10 token (quá ngắn, thiếu ngữ cảnh)

def analyze_json(file_path: Path):
    if not file_path.exists():
        return None
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    stats = {
        "pipeline": data.get("pipeline", "unknown"),
        "total_chunks_db": 0,    # Đây là số chunk thực sự nạp vào VectorDB
        "total_parents": 0,
        "total_children": 0,
        "over_max": 0,
        "under_min": 0,
        "max_tokens": 0,
        "min_tokens": float('inf'),
        "max_token_source": "",
        "min_token_source": "",
    }
    
    chunks = data.get("chunks", [])
    
    # Flatten children (do JSON của mình lưu children lồng trong parent)
    all_chunks = []
    for c in chunks:
        if c.get("chunk_level") == "parent":
            all_chunks.append(c)
            all_chunks.extend(c.get("children", []))
        elif c.get("chunk_level") == "child":
            # Just in case json format changes
            all_chunks.append(c)
        else:
            all_chunks.append(c)
            
    # Bắt đầu phân tích
    stats["total_chunks_db"] = len(all_chunks)
    
    for c in all_chunks:
        level = c.get("chunk_level", "unknown")
        if level == "parent":
            stats["total_parents"] += 1
        elif level == "child":
            stats["total_children"] += 1
            
        token_count = c.get("token_count", 0)
        source = c.get("source", "unknown")
        section = c.get("section_name", "unknown")[:30]
        
        # Min / Max tracking
        if token_count > stats["max_tokens"]:
            stats["max_tokens"] = token_count
            stats["max_token_source"] = f"{source} ({section}...)"
            
        if token_count < stats["min_tokens"]:
            stats["min_tokens"] = token_count
            stats["min_token_source"] = f"{source} ({section}...)"
            
        # Warning tracking
        if token_count > MAX_TOKEN_WARN:
            stats["over_max"] += 1
        if token_count < MIN_TOKEN_WARN:
            stats["under_min"] += 1
            
    if stats["min_tokens"] == float('inf'):
        stats["min_tokens"] = 0
            
    return stats

def print_report(name, stats):
    print(f"\n{'═'*50}")
    print(f"📊 BÁO CÁO PIPELINE: {name.upper()}")
    print(f"{'═'*50}")
    
    if not stats:
        print("❌ Không tìm thấy file dữ liệu. Hãy chạy export_chunks.py trước.")
        return
        
    print(f"📦 TỔNG CHUNK VÀO DB : {stats['total_chunks_db']} chunks")
    print(f"   ├─ Số lượng Parent: {stats['total_parents']} khối")
    print(f"   └─ Số lượng Child : {stats['total_children']} khối")
    print()
    print(f"⚠️  CẢNH BÁO TOKEN (Mức lý tưởng: 10 - 800 tokens):")
    print(f"   ├─ Vượt quá {MAX_TOKEN_WARN} token : {stats['over_max']} chunk(s) 🛑")
    print(f"   └─ Quá ngắn < {MIN_TOKEN_WARN} token: {stats['under_min']} chunk(s) 🟡")
    print()
    print(f"📏 THỐNG KÊ CHIỀU DÀI:")
    print(f"   ├─ Dài nhất (MAX) : {stats['max_tokens']} tokens")
    print(f"   │    └─ Tại: {stats['max_token_source']}")
    print(f"   └─ Ngắn nhất (MIN): {stats['min_tokens']} tokens")
    print(f"   │    └─ Tại: {stats['min_token_source']}")

# --- MAIN ---
print("\n🔍 ĐANG TIẾN HÀNH PHÂN TÍCH TOÀN BỘ KHO CHUNK...")
path_markdown = Path("data/exports/chunks_raw_markdown.json")
path_structured = Path("data/exports/chunks_raw_structured.json")

stats_md = analyze_json(path_markdown)
stats_st = analyze_json(path_structured)

print_report("Markdown (Thông báo chung)", stats_md)
print_report("Structured (Chương trình đào tạo)", stats_st)

print(f"\n{'═'*50}")
total_db = 0
if stats_md: total_db += stats_md['total_chunks_db']
if stats_st: total_db += stats_st['total_chunks_db']
print(f"🚀 TÔNG SỐ VECTOR SẼ NẠP VÀO POSTGRES: {total_db} VECTORS")
print(f"{'═'*50}\n")
