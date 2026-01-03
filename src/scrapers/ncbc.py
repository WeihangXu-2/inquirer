import os, re, time, hashlib
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from urllib.parse import urljoin

USER_AGENT = "nc-civil-ai/0.1 (research; contact: you@example.com)"

BASE_DOMAIN = "https://www.nccourts.gov"

def absolutize(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    # If it's already absolute, keep it; else join with site root
    return u if u.startswith(("http://", "https://")) else urljoin(BASE_DOMAIN, u)

class NCBCOpinion(BaseModel):
    court: str = "NC Business Court"
    ncbc_cite: Optional[str] = None
    case_title: Optional[str] = None
    case_number: Optional[str] = None
    county: Optional[str] = None
    judge: Optional[str] = None
    published_date: Optional[str] = None   # YYYY-MM-DD
    pdf_url: Optional[str] = None

def _get(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text

def fetch_listing_page(page: int, base_url: str, timeout: int = 30) -> str:
    # NCBC pagination uses ?page=0 for page 1, ?page=1 for page 2, etc.
    url = f"{base_url}?page={page-1}" if page > 1 else base_url
    return _get(url, timeout=timeout)

def parse_listing(html: str) -> List[NCBCOpinion]:
    from datetime import datetime
    soup = BeautifulSoup(html, "html.parser")
    opinions: List[NCBCOpinion] = []

    for h in soup.find_all(["h4", "h5"]):
        title = h.get_text(strip=True)
        if not title:
            continue

        # Try to find NCBC citation within the title
        ncbc_cite = None
        m = re.search(r"(20\d{2}\s+NCBC\s+\d+)", title)
        if m:
            ncbc_cite = m.group(1)

        # Find the nearest "view/download" link under this heading
        pdf_url = None
        for a in h.find_all_next("a"):
            txt = (a.get_text() or "").lower()
            if "view/download" in txt and a.has_attr("href"):
                # make sure it's the link associated with this h, not a future item
                prev_heading = a.find_previous(["h4","h5"])
                if prev_heading == h:
                    pdf_url = a["href"]
                    break

        # Try to parse the metadata line that usually sits before the heading
        meta_node = h.find_previous(string=True)
        meta = (meta_node or "").strip()
        # date like "August 15, 2025"
        published_date = None
        md = re.search(r"([A-Za-z]+\s+\d{1,2},\s+\d{4})", meta)
        if md:
            try:
                published_date = datetime.strptime(md.group(1), "%B %d, %Y").date().isoformat()
            except Exception:
                published_date = None

        # case number e.g. 23-CVS-581
        case_number = None
        m2 = re.search(r"(\d{2,}[-–—]?CVS[-–—]?\d+)", meta, flags=re.I)
        if m2:
            case_number = m2.group(1).replace("–","-").replace("—","-")

        # county and judge inside parentheses: (Wake - Judge Name)
        county = None
        judge = None
        m3 = re.search(r"\(([^()]+?)\s*-\s*([^()]+?)\)", meta)
        if m3:
            county = m3.group(1).strip()
            judge  = m3.group(2).strip()

        opinions.append(NCBCOpinion(
            ncbc_cite=ncbc_cite,
            case_title=title,
            case_number=case_number,
            county=county,
            judge=judge,
            published_date=published_date,
            pdf_url=pdf_url
        ))

    return opinions

def download_pdf(url: Optional[str], out_dir: str, sleep_sec: float = 1.2) -> Optional[str]:
    if not url:
        return None
    url = absolutize(url)
    os.makedirs(out_dir, exist_ok=True)
    fname = hashlib.sha256(url.encode()).hexdigest() + ".pdf"
    path = os.path.join(out_dir, fname)
    if os.path.exists(path):
        return path

    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<15):
                if chunk:
                    f.write(chunk)
    time.sleep(sleep_sec)  # be polite
    return path
