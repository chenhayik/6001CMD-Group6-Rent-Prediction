"""Count the words in the main text of a .docx report.

Rules: exclude cover page, table of contents, tables, figures,
captions, references, and appendices.
Usage: python wordcount.py ["6001CMD_GroupProject_Group6.docx"]
If you give no file name, the script uses DEFAULT_FILE. It looks in
the current folder, the script folder, and submission_forms/.
If the file is not a valid .docx (for example, Word or OneDrive is
saving it), the script tells you why and uses the newest valid copy.
"""
import re
import sys
import time
import zipfile
from pathlib import Path

import docx

DEFAULT_FILE = "6001CMD_GroupProject_Group6.docx"
SKIP = re.compile(r"reference|appendi|bibliograph|content", re.I)
CAPTION = re.compile(r"^(Figure|Table|Listing)\s+\d+(\.\d+)*[:.]?\s")
MAX_CAPTION_WORDS = 25
HERE = Path(__file__).resolve().parent
FOLDERS = (Path.cwd(), HERE, HERE / "submission_forms")


def find_file(name):
    path = Path(name)
    if path.is_absolute():
        return path
    for folder in FOLDERS:
        if (folder / name).exists():
            return folder / name
    return path


def is_valid_docx(path, tries=3):
    """Return (True, "") or (False, reason). Retry while a save runs."""
    reason = ""
    for attempt in range(tries):
        try:
            if not path.exists():
                return False, "the file does not exist"
            size = path.stat().st_size
            if size == 0:
                reason = "the file is empty (0 bytes)"
            elif zipfile.is_zipfile(path):
                return True, ""
            else:
                reason = f"the file ({size:,} bytes) is not a valid .docx"
        except PermissionError:
            reason = "Windows does not let Python read the file"
        if attempt < tries - 1:
            time.sleep(2)  # Word or OneDrive can be in the middle of a save
    lock = path.with_name("~$" + path.name[2:])
    if lock.exists():
        reason += "; Word has the file open (a ~$ lock file exists)"
    return False, reason


def find_fallback(bad_path):
    """Find the newest valid .docx with the same name start."""
    stem = Path(DEFAULT_FILE).stem
    found = set()
    for folder in FOLDERS:
        if folder.exists():
            found.update(folder.glob(stem + "*.docx"))
    good = [p for p in found
            if not p.name.startswith("~$")
            and p.resolve() != bad_path.resolve()
            and is_valid_docx(p, tries=1)[0]]
    return max(good, key=lambda p: p.stat().st_mtime) if good else None


def is_caption(style, text):
    if style.lower().startswith("caption"):
        return True
    # A caption is a short label with no full stop at the end.
    # A body sentence such as "Table 1.2 lists ..." ends with a full stop.
    return (bool(CAPTION.match(text + " "))
            and len(text.split()) <= MAX_CAPTION_WORDS
            and not text.endswith("."))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE
    path = find_file(name)
    ok, reason = is_valid_docx(path)
    if not ok:
        print(f"Cannot open: {path}")
        print(f"Reason: {reason}.")
        print("Do these steps:")
        print("  1. Save and close the file in Word.")
        print("  2. Wait until OneDrive shows a green tick on the file.")
        print('  3. In File Explorer, right-click the file and select'
              ' "Always keep on this device".')
        print("  4. Run the script again.")
        path = find_fallback(path)
        if path is None:
            print("No other valid copy found. Stop.")
            sys.exit(1)
        print(f"\nWARNING: This count uses another copy: {path}")
        print("This copy can be older than your latest changes.\n")
    print(f"File: {path}")
    doc = docx.Document(str(path))
    count_on = False
    words = 0
    per_chapter = {}
    chapter = None
    # doc.paragraphs does not include table text, so tables are excluded.
    for p in doc.paragraphs:
        text = p.text.strip()
        style = p.style.name
        if style.startswith("Heading 1"):
            if not text:
                continue  # ignore empty Heading 1 lines
            count_on = not SKIP.search(text)
            chapter = text
        if not count_on or not text:
            continue
        if is_caption(style, text):
            continue
        n = len(text.split())
        words += n
        per_chapter[chapter] = per_chapter.get(chapter, 0) + n
    for chapter_name, n in per_chapter.items():
        print(f"{n:6d}  {chapter_name}")
    print(f"Total words: {words}")
    print("Range: 4000 to 6000")
    print("OK" if 4000 <= words <= 6000 else "OUT OF RANGE")


if __name__ == "__main__":
    main()
