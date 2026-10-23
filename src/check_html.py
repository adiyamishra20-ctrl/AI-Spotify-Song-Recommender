from pathlib import Path
from bs4 import BeautifulSoup

project_root = Path(__file__).resolve().parents[1]
page = project_root / "data" / "raw" / "yellow.html"

html = page.read_text(encoding="utf-8")
soup = BeautifulSoup(html, "html.parser")

print("HTML length:", len(html))
print("Page title:", soup.title.get_text(strip=True) if soup.title else "Missing")

print("\nFirst 1,500 characters of visible text:")
print(soup.get_text(" ", strip=True)[:1500])

print("\nFirst 500 characters of HTML:")
print(html[:500])