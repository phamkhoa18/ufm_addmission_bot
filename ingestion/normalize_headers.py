"""
Normalize Headers — Chuyển đổi Legacy text header → YAML Frontmatter chuẩn.

Script quét file Markdown, nhận dạng header cũ (Ngày hiệu lực:...),
tự động sinh YAML Frontmatter chuẩn và ghi lại file (hoặc preview).

Sử dụng:
  # Preview tất cả file (KHÔNG ghi đè)
  python ingestion/normalize_headers.py --dry-run

  # Chuẩn hóa 1 file cụ thể
  python ingestion/normalize_headers.py --file data/unstructured/markdown/maudon/dondangkitiensi.md

  # Chuẩn hóa tất cả file trong thư mục chỉ định
  python ingestion/normalize_headers.py --dir data/unstructured/markdown/thongtinchung

  # Chuẩn hóa TẤT CẢ thư mục (thongtinchung + maudon)
  python ingestion/normalize_headers.py --all

  # Ghi đè file gốc (CẢNH BÁO: tạo backup trước)
  python ingestion/normalize_headers.py --all --write

  # Chỉ kiểm tra file nào đã chuẩn, file nào chưa
  python ingestion/normalize_headers.py --all --check-only
"""

import argparse
import io
import os
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

# Fix Windows terminal encoding (cp1258 khong ho tro day du tieng Viet)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -- Dam bao import path --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Thư mục data
DATA_ROOT = PROJECT_ROOT / "data" / "unstructured" / "markdown"
BACKUP_SUFFIX = ".bak"

# Mapping thư mục → doc_type
DIR_TO_DOCTYPE = {
    "thongtinchung": "thongtinchung",
    "maudon": "maudon",
}


# ================================================================
# SLUGIFY — Chuẩn hóa tên file → doc_id (không dấu, lowercase)
# ================================================================
def slugify(text: str) -> str:
    """
    Chuyển chuỗi tiếng Việt thành slug ASCII an toàn.

    "Tuyển sinh trình độ tiến sĩ đợt 1" → "tuyen-sinh-trinh-do-tien-si-dot-1"
    "Giấy cam đoan" → "giay-cam-doan"
    "mẫu ĐƠN DỰ TUYỂN" → "mau-don-du-tuyen"
    """
    # Lowercase
    text = text.lower().strip()

    # Bảng chuyển đổi tiếng Việt → ASCII
    vietnamese_map = {
        "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
        "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
        "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
        "đ": "d",
        "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
        "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
        "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
        "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
        "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
        "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
        "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
        "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
        "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    }

    result = []
    for ch in text:
        if ch in vietnamese_map:
            result.append(vietnamese_map[ch])
        elif ch.isascii() and ch.isalnum():
            result.append(ch)
        elif ch in (" ", "_", "-", "."):
            result.append("-")
        # Bỏ qua ký tự đặc biệt khác

    slug = "".join(result)
    # Gộp nhiều dấu gạch ngang liên tiếp
    slug = re.sub(r"-{2,}", "-", slug)
    # Xóa đầu/cuối
    slug = slug.strip("-")
    return slug


# ================================================================
# CLEAN MARKDOWN ESCAPES — Xóa backslash escape thừa
# ================================================================
def clean_markdown_escapes(text: str) -> str:
    """
    Xoa backslash escape thua do tool chuyen doi Word/PDF -> Markdown tao ra.

    Cac tool nhu Pandoc, Markdownify tu dong them '\\' truoc cac ky tu
    dac biet cua Markdown (-, ., [, ], *) de "an toan", nhung gay ra:
      - '\\-start-' khong duoc nhan dang boi header_parser
      - '## 1\\. Nganh' render xau tren Frontend
      - '\\[**]' thanh ky tu vo nghia

    Ham nay strip cac backslash escape pho bien:
      '\\-'  -> '-'   (bullet list escape)
      '\\.'  -> '.'   (ordered list escape)
      '\\['  -> '['   (link/image escape)
      '\\]'  -> ']'
      '\\*'  -> '*'   (bold/italic escape)
      '\\_'  -> '_'   (underscore escape)
    """
    # Chi strip backslash TRUOC cac ky tu Markdown dac biet
    # KHONG strip backslash truoc ky tu khac (co the la escape hop le)
    text = re.sub(r'\\([-.*\[\]_])', r'\1', text)
    return text


# ================================================================
# DETECT FORMAT — Nhan dang file da chuan hay chua
# ================================================================
def detect_format(content: str) -> str:
    """
    Tra ve:
      "yaml"   — Da co YAML Frontmatter (chuan moi)
      "legacy" — Header text cu (Ngay hieu luc:...)
      "none"   — Khong co header nao
    """
    if content.strip().startswith("---"):
        return "yaml"
    # Kiem tra ca dang co backslash escape: \-start-
    if "-start-" in content or "\\-start-" in content:
        return "legacy"
    return "none"


# ================================================================
# PARSE LEGACY HEADER → Trích xuất metadata
# ================================================================
def extract_legacy_metadata(content: str, filepath: Path) -> dict:
    """
    Đọc header text cũ và trích xuất metadata.

    Returns:
        dict với các key: doc_type, doc_id, title, effective_date,
                          doc_number, program_level, academic_year, ...
    """
    meta = {
        "doc_type": None,
        "doc_id": None,
        "title": None,
        "effective_date": None,
        "doc_number": None,
        "program_level": None,
        "academic_year": None,
        "parent_doc_id": None,
        "keywords": [],
    }

    # Xác định doc_type từ thư mục cha
    parent_dir = filepath.parent.name
    meta["doc_type"] = DIR_TO_DOCTYPE.get(parent_dir, "thongtinchung")

    # doc_id từ tên file (slugify)
    stem = filepath.stem  # Tên file không đuôi
    meta["doc_id"] = slugify(stem)

    # Tách header
    if "-start-" not in content:
        return meta

    header = content.split("-start-", 1)[0].strip()

    # ── Ngày hiệu lực ──
    date_match = re.search(r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})", header)
    if date_match:
        try:
            dt = datetime.strptime(date_match.group(1), "%d/%m/%Y")
            meta["effective_date"] = dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    # ── Văn bản số ──
    doc_num_match = re.search(r"Văn bản\s*[Ss]ố:\s*(.+)", header)
    if doc_num_match:
        raw = doc_num_match.group(1).strip()
        # Xóa khoảng trắng thừa quanh /
        raw = re.sub(r"\s*/\s*", "/", raw)
        meta["doc_number"] = raw

    # ── Title: lấy từ heading # lớn nhất hoặc dòng ## đầu tiên ──
    title_candidates = []
    for line in header.splitlines():
        stripped = line.strip().lstrip("#").strip().strip("*").strip()
        if stripped and not stripped.lower().startswith("ngày hiệu lực"):
            if not stripped.lower().startswith("văn bản"):
                title_candidates.append(stripped)

    # Ưu tiên heading có chứa tên mẫu đơn cụ thể
    for candidate in title_candidates:
        lower_c = candidate.lower()
        if any(kw in lower_c for kw in [
            "mẫu đơn", "đơn đăng ký", "đơn dự tuyển", "giấy cam đoan",
            "đề cương", "phiếu đăng ký", "lý lịch", "phụ lục"
        ]):
            meta["title"] = candidate
            break

    if not meta["title"] and title_candidates:
        meta["title"] = title_candidates[-1]  # Lấy dòng cuối cùng (thường cụ thể nhất)

    # ── Program level ──
    header_lower = header.lower()
    level_patterns = [
        (r"trình\s+độ\s+tiến\s+sĩ", "tien_si"),
        (r"trình\s+độ\s+thạc\s+sĩ", "thac_si"),
        (r"tiến\s+sĩ", "tien_si"),
        (r"thạc\s+sĩ", "thac_si"),
        (r"đại\s+học", "dai_hoc"),
    ]
    for pattern, level in level_patterns:
        if re.search(pattern, header_lower):
            meta["program_level"] = level
            break

    # ── Academic year ──
    year_match = re.search(r"năm\s+(\d{4})", header_lower)
    if year_match:
        meta["academic_year"] = year_match.group(1)

    # ── Keywords tự động từ program_level ──
    if meta["program_level"] == "thac_si":
        meta["keywords"] = ["thạc sĩ", "cao học", "tuyển sinh"]
    elif meta["program_level"] == "tien_si":
        meta["keywords"] = ["tiến sĩ", "NCS", "nghiên cứu sinh"]
    elif meta["program_level"] == "dai_hoc":
        meta["keywords"] = ["đại học", "tuyển sinh"]

    return meta


# ================================================================
# BUILD YAML FRONTMATTER
# ================================================================
def build_frontmatter(meta: dict) -> str:
    """Sinh YAML Frontmatter chuẩn từ metadata dict."""
    lines = ["---"]

    # Required
    lines.append(f'doc_type: {meta["doc_type"]}')
    lines.append(f'doc_id: {meta["doc_id"]}')
    if meta["title"]:
        # Escape dấu nháy kép trong title
        safe_title = meta["title"].replace('"', '\\"')
        lines.append(f'title: "{safe_title}"')
    else:
        lines.append(f'title: "{meta["doc_id"]}"')

    if meta["effective_date"]:
        lines.append(f'effective_date: {meta["effective_date"]}')

    # Optional
    if meta.get("doc_number"):
        lines.append(f'doc_number: "{meta["doc_number"]}"')

    if meta.get("program_level"):
        lines.append(f'program_level: {meta["program_level"]}')

    if meta.get("academic_year"):
        lines.append(f'academic_year: "{meta["academic_year"]}"')

    if meta.get("parent_doc_id"):
        lines.append(f'parent_doc_id: {meta["parent_doc_id"]}')

    if meta.get("keywords"):
        kw_str = ", ".join(meta["keywords"])
        lines.append(f"keywords: [{kw_str}]")

    lines.append("---")
    return "\n".join(lines)


# ================================================================
# CONVERT FILE
# ================================================================
def convert_file(filepath: Path, write: bool = False) -> dict:
    """
    Chuyển đổi 1 file từ Legacy header → YAML Frontmatter.

    Returns:
        dict: {
            "file": str,
            "status": "converted" | "already_yaml" | "no_header" | "error",
            "meta": dict | None,
            "preview": str | None,
        }
    """
    try:
        raw = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return {"file": str(filepath), "status": "error", "meta": None, "preview": str(e)}

    # Buoc 0: Clean backslash escape truoc khi lam bat ky gi
    # (\-start- -> -start-, ## 1\. -> ## 1., ...)
    raw = clean_markdown_escapes(raw)

    fmt = detect_format(raw)

    if fmt == "yaml":
        return {"file": str(filepath.name), "status": "already_yaml", "meta": None, "preview": None}

    if fmt == "none":
        return {"file": str(filepath.name), "status": "no_header", "meta": None, "preview": None}

    # -- Legacy -> Convert --
    meta = extract_legacy_metadata(raw, filepath)

    # Tach phan body (sau -start-)
    parts = raw.split("-start-", 1)
    header_text = parts[0].strip()
    body_text = parts[1] if len(parts) > 1 else ""

    # Xử lý phần giữa frontmatter và -start- (giữ lại làm header_context)
    # Loại bỏ dòng "Ngày hiệu lực" và "Văn bản số" (đã nạp vào YAML)
    context_lines = []
    for line in header_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^Ngày hiệu lực:", stripped, re.IGNORECASE):
            continue
        if re.match(r"^Văn bản\s*[Ss]ố:", stripped, re.IGNORECASE):
            continue
        context_lines.append(line)

    between_text = "\n".join(context_lines).strip()

    # Build file mới
    frontmatter = build_frontmatter(meta)
    new_parts = [frontmatter]
    if between_text:
        new_parts.append(between_text)
    new_parts.append("-start-")
    new_content = "\n".join(new_parts) + body_text

    if write:
        # Tạo backup
        backup_path = filepath.with_suffix(filepath.suffix + BACKUP_SUFFIX)
        if not backup_path.exists():
            filepath.rename(backup_path)
            # Ghi file mới
            filepath.write_text(new_content, encoding="utf-8")
        else:
            # Backup đã tồn tại → ghi đè file gốc
            filepath.write_text(new_content, encoding="utf-8")

    # Preview: chỉ lấy phần đầu (frontmatter + vài dòng)
    preview_lines = new_content.split("\n")[:20]
    preview = "\n".join(preview_lines)
    if len(new_content.split("\n")) > 20:
        preview += "\n... (còn tiếp)"

    return {
        "file": str(filepath.name),
        "status": "converted",
        "meta": meta,
        "preview": preview,
    }


# ================================================================
# LUỒNG CHÍNH
# ================================================================
def collect_files(
    single_file: str = None,
    target_dir: str = None,
    scan_all: bool = False,
) -> list[Path]:
    """Thu thập danh sách file .md cần xử lý."""
    if single_file:
        p = Path(single_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"File khong ton tai: {p}")
        return [p]

    if target_dir:
        d = Path(target_dir)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        if not d.exists():
            raise FileNotFoundError(f"Thu muc khong ton tai: {d}")
        return sorted(d.rglob("*.md"))

    if scan_all:
        files = []
        for subdir in ["thongtinchung", "maudon"]:
            d = DATA_ROOT / subdir
            if d.exists():
                files.extend(sorted(d.rglob("*.md")))
        return files

    return []


def run(
    single_file: str = None,
    target_dir: str = None,
    scan_all: bool = False,
    write: bool = False,
    check_only: bool = False,
):
    """Pipeline chính."""
    print("=" * 65)
    print("  UFM - Normalize Headers to YAML Frontmatter")
    print("  Thoi gian: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  Mode: %s" % ("WRITE (ghi de file)" if write else "DRY-RUN (chi xem truoc)"))
    print("=" * 65)

    files = collect_files(single_file, target_dir, scan_all)

    if not files:
        print("\n[X] Khong tim thay file nao! Dung --file, --dir hoac --all.")
        return

    print("\n[i] Tim thay %d file:" % len(files))
    for f in files:
        print("   - %s" % f.relative_to(PROJECT_ROOT))

    # Phan loai
    results = {"converted": [], "already_yaml": [], "no_header": [], "error": []}

    print("\n" + "-" * 65)

    for filepath in files:
        # Bo qua file backup
        if filepath.name.endswith(BACKUP_SUFFIX):
            continue
        # Bo qua file tam
        if filepath.name.startswith("~$"):
            continue

        if check_only:
            raw = filepath.read_text(encoding="utf-8")
            # Clean escapes truoc khi detect
            cleaned = clean_markdown_escapes(raw)
            fmt = detect_format(cleaned)
            status_icon = {"yaml": "[OK]", "legacy": "[!!]", "none": "[??]"}
            status_text = {"yaml": "YAML (chuan)", "legacy": "LEGACY (can chuyen)", "none": "Khong co header"}
            print("  %s %-40s [%s]" % (status_icon.get(fmt, "?"), filepath.name, status_text.get(fmt, "?")))
            results[{"yaml": "already_yaml", "legacy": "converted", "none": "no_header"}.get(fmt, "error")].append(filepath.name)
            continue

        result = convert_file(filepath, write=write)
        results[result["status"]].append(result)

        icon = {"converted": "[>>]", "already_yaml": "[OK]", "no_header": "[??]", "error": "[XX]"}
        print("\n  %s %s" % (icon.get(result["status"], "?"), result["file"]))
        print("     Status: %s" % result["status"])

        if result["status"] == "converted":
            meta = result["meta"]
            print("     doc_type: %s" % meta["doc_type"])
            print("     doc_id:   %s" % meta["doc_id"])
            print("     title:    %s" % meta.get("title", "N/A"))
            print("     date:     %s" % meta.get("effective_date", "N/A"))
            print("     level:    %s" % meta.get("program_level", "N/A"))

            if not write:
                print("\n     PREVIEW:")
                for line in (result["preview"] or "").split("\n"):
                    print("        %s" % line)

    # Tong ket
    print("\n" + "=" * 65)
    print("TONG KET:")
    print("   [>>] Converted:    %d" % len(results["converted"]))
    print("   [OK] Already YAML: %d" % len(results["already_yaml"]))
    print("   [??] No header:    %d" % len(results["no_header"]))
    print("   [XX] Error:        %d" % len(results["error"]))

    if not write and results["converted"]:
        print("\n[TIP] De ghi de file, them flag --write")
        print("      File goc se duoc backup thanh *.md.bak")
    print("=" * 65)


# ================================================================
# COMPRESS FORM PLACEHOLDERS — Nen ky tu .... vo nghia trong mau don
# ================================================================
def compress_form_placeholders(text: str) -> str:
    """
    Nen cac ky tu filler vo nghia trong mau don hanh chinh.

    TRUOC (lang phi ~200 tokens):
      Toi ten la:....................
      Sinh ngay:.......................... tai:.......................
      ...................................................................................................................................................
      Ngay ..... thang ..... nam 20......

    SAU (tiet kiem, cau truc 100% giu nguyen):
      Toi ten la: ___
      Sinh ngay: ___ tai: ___
      (dong chi toan dau cham -> xoa)
      Ngay ___ thang ___ nam 20___

    Quy tac:
      1. Dong CHI co dau cham + khoang trang -> xoa hoan toan
      2. 3+ dau cham lien tiep -> ___
      3. Don nhieu khoang trang lien tiep ve 1
      4. KHONG thay doi YAML frontmatter (giu nguyen phan --- ... ---)
    """
    lines = text.split("\n")
    result_lines = []
    in_frontmatter = False
    frontmatter_count = 0

    for line in lines:
        # Bao ve YAML frontmatter — khong nen
        stripped = line.strip()
        if stripped == "---":
            frontmatter_count += 1
            if frontmatter_count == 1:
                in_frontmatter = True
            elif frontmatter_count == 2:
                in_frontmatter = False
            result_lines.append(line)
            continue

        if in_frontmatter:
            result_lines.append(line)
            continue

        # Rule 1: Dong CHI co dau cham va khoang trang -> xoa
        if stripped and all(c in ".… " for c in stripped):
            continue

        # Rule 2: 3+ dau cham lien tiep -> ___
        compressed = re.sub(r"\.{3,}", "___", line)

        # Rule 2b: 3+ dau cham Unicode (…) -> ___
        compressed = re.sub(r"…+", "___", compressed)

        # Rule 3: Sap xep lai khoang trang quanh placeholder
        # "ten la:___" -> "ten la: ___"
        compressed = re.sub(r":(\s*)___", ": ___", compressed)
        # "___tai:" -> "___ tai:"
        compressed = re.sub(r"___(\s*)(\S)", r"___ \2", compressed)

        # Rule 4: Don nhieu khoang trang lien tiep
        compressed = re.sub(r"  +", " ", compressed)

        # Rule 5: Xoa trailing whitespace
        compressed = compressed.rstrip()

        result_lines.append(compressed)

    # Xoa cac dong trong lien tiep (nhieu hon 2 dong trong -> giu 1)
    final_lines = []
    blank_count = 0
    for line in result_lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                final_lines.append(line)
        else:
            blank_count = 0
            final_lines.append(line)

    return "\n".join(final_lines)


def run_compress_forms():
    """Nen tat ca file mau don trong thu muc maudon."""
    maudon_dir = DATA_ROOT / "maudon"
    if not maudon_dir.exists():
        print("[X] Thu muc maudon khong ton tai: %s" % maudon_dir)
        return

    files = sorted(maudon_dir.glob("*.md"))
    if not files:
        print("[X] Khong tim thay file .md nao trong %s" % maudon_dir)
        return

    print("=" * 65)
    print("  UFM - Compress Form Template Placeholders")
    print("  Thoi gian: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65)
    print("\n[i] Tim thay %d file mau don:" % len(files))

    total_saved = 0

    for filepath in files:
        print("\n  --- %s ---" % filepath.name)

        raw = filepath.read_text(encoding="utf-8")
        original_len = len(raw)

        compressed = compress_form_placeholders(raw)
        new_len = len(compressed)
        saved = original_len - new_len
        saved_pct = (saved * 100 // original_len) if original_len > 0 else 0

        # Dem so luong dot chars truoc/sau
        dots_before = sum(len(m) for m in re.findall(r"\.{3,}", raw))
        dots_after = sum(len(m) for m in re.findall(r"\.{3,}", compressed))

        print("     Truoc: %d chars, %d filler dots" % (original_len, dots_before))
        print("     Sau:   %d chars, %d filler dots" % (new_len, dots_after))
        print("     Tiet kiem: %d chars (%d%%)" % (saved, saved_pct))

        if saved > 0:
            # Backup
            backup_path = filepath.with_suffix(filepath.suffix + ".bak")
            if not backup_path.exists():
                import shutil
                shutil.copy2(filepath, backup_path)
                print("     [BACKUP] %s" % backup_path.name)

            # Ghi de
            filepath.write_text(compressed, encoding="utf-8")
            print("     [OK] Da ghi de file")
            total_saved += saved
        else:
            print("     [--] File da sach, khong can nen")

    print("\n" + "=" * 65)
    print("TONG KET:")
    print("   Tong tiet kiem: %d chars (~%d tokens)" % (total_saved, total_saved // 4))
    print("   Cac file goc da duoc backup thanh *.md.bak")
    print("=" * 65)


# ================================================================
# CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Normalize Markdown headers to YAML Frontmatter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Vi du:
  python ingestion/normalize_headers.py --all --check-only
  python ingestion/normalize_headers.py --file data/unstructured/markdown/maudon/dondangkitiensi.md
  python ingestion/normalize_headers.py --dir data/unstructured/markdown/thongtinchung
  python ingestion/normalize_headers.py --all
  python ingestion/normalize_headers.py --all --write
  python ingestion/normalize_headers.py --compress-forms
        """,
    )
    parser.add_argument("--file", type=str, help="Xu ly 1 file cu the")
    parser.add_argument("--dir", type=str, help="Xu ly tat ca file trong thu muc")
    parser.add_argument("--all", action="store_true", help="Xu ly tat ca (thongtinchung + maudon)")
    parser.add_argument("--write", action="store_true", help="Ghi de file goc (tao backup .bak)")
    parser.add_argument("--check-only", action="store_true", help="Chi kiem tra trang thai, khong chuyen doi")
    parser.add_argument("--compress-forms", action="store_true",
                        help="Nen ky tu .... vo nghia trong mau don (maudon/)")

    args = parser.parse_args()

    if args.compress_forms:
        run_compress_forms()
        return

    if not any([args.file, args.dir, args.all]):
        parser.print_help()
        return

    run(
        single_file=args.file,
        target_dir=args.dir,
        scan_all=args.all,
        write=args.write,
        check_only=args.check_only,
    )


if __name__ == "__main__":
    main()

