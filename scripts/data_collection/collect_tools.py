#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect AI-agent tool metadata from LangChain docs, CrewAI/LlamaIndex GitHub repos, and npm (n8n nodes).
Output: JSONL (one tool per line, unified schema). Python >=3.9.

Ethical use: Intended for research and non-commercial use. Respect each source's terms of service and
rate limits. Use GITHUB_TOKEN when possible to reduce load on public APIs. The script uses polite
delays between requests; do not lower them to avoid overloading servers.

Usage: python collect_tools.py --out tools.jsonl --sources langchain crewai llamaindex n8n --github-token $GITHUB_TOKEN --npm-pages 8
Tip: Export GITHUB_TOKEN for higher GitHub API rate limit. For first n8n crawl use --npm-pages 8~12 (size=250 per page).
"""
import argparse
import os
import re
import sys
import time
import json
import tarfile
import tempfile
import shutil
import random
import csv
from pathlib import Path
from urllib.parse import urljoin, urlparse, quote
from datetime import datetime, timezone
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

HEADERS = {
    "User-Agent": "ToolCrawler/1.0 (research; respect ToS and rate limits)"
}
MIN_DELAY_BETWEEN_REQUESTS = 0.35
LANGCHAIN_INDEX = "https://docs.langchain.com/oss/python/integrations/tools"
LANGCHAIN_BASE = "https://docs.langchain.com"
GITHUB_API_DIR = "https://api.github.com/repos/crewAIInc/crewAI-tools/contents/crewai_tools"
NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
NPM_PKG_BASE = "https://registry.npmjs.org"
DOWNLOADS_URL = "https://api.npmjs.org/downloads/point/last-week/{name}"
N8N_OFFICIAL_OWNER = "n8n-io"
N8N_OFFICIAL_REPO  = "n8n"
N8N_OFFICIAL_PATH  = "packages/nodes-base/nodes"
LLAMAINDEX_OWNER = "run-llama"
LLAMAINDEX_REPO = "llama_index"
LLAMAINDEX_TOOLS_PATH = "llama-index-integrations/tools"

EXPORT_TS = datetime.now(timezone.utc)


class SoftError(Exception):
    pass

@retry(reraise=True, stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8),
       retry=retry_if_exception_type((requests.RequestException, SoftError)))
def _get(session: requests.Session, url: str, **kwargs):
    resp = session.get(url, headers=HEADERS, timeout=30, **kwargs)
    if resp.status_code >= 500:
        raise SoftError(f"5xx from {url}")
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise SoftError(f"Rate limited: {url}")
    resp.raise_for_status()
    return resp

def write_jsonl(path: Path, items):
    with path.open("a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def to_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\-_.]+", "-", s.strip()).strip("-").lower()
    return s[:120] if len(s) > 120 else s

def safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def ensure_set(x):
    if x is None: return set()
    if isinstance(x, (list, tuple, set)): return set(x)
    return {x}

def normalize_repo(u: str) -> str:
    if not u: return ""
    u = u.strip()
    u = re.sub(r'^git\+', '', u)
    u = re.sub(r'\.git$', '', u)
    m = re.match(r'^git@github\.com:(.+)$', u)
    if m:
        return f"https://github.com/{m.group(1)}"
    return u

def safe_pkg(name: str) -> str:
    return quote(name, safe='')

def get_repo_home_from_registry(session: requests.Session, name: str):
    try:
        url = f"{NPM_PKG_BASE}/{safe_pkg(name)}"
        r = _get(session, url)
        j = r.json()
        dist_tags = j.get("dist-tags", {})
        latest = dist_tags.get("latest")
        meta = j.get("versions", {}).get(latest, {}) if latest else {}
        repo = meta.get("repository", {})
        repo_url = repo.get("url") if isinstance(repo, dict) else (repo or "")
        homepage = meta.get("homepage") or ""
        return normalize_repo(repo_url), homepage
    except Exception as e:
        return "", ""

def get_weekly_downloads(session: requests.Session, name: str, max_attempts=8):
    url = DOWNLOADS_URL.format(name=safe_pkg(name))
    attempt = 0
    while attempt < max_attempts:
        try:
            r = session.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else min(60, 2**attempt)
                time.sleep(wait + random.uniform(0, 0.3))
                attempt += 1
                continue
            r.raise_for_status()
            return int(r.json().get("downloads", 0))
        except requests.RequestException:
            wait = min(60, 2**attempt)
            time.sleep(wait + random.uniform(0, 0.3))
            attempt += 1
    return 0

def get_json_with_retry(session: requests.Session, url: str, params=None, attempt=0, max_attempts=8, headers=None):
    """Enhanced GET with better 429 and 5xx handling."""
    headers = headers or HEADERS
    try:
        r = session.get(url, params=params, headers=headers, timeout=30)
    except requests.RequestException as e:
        if attempt >= max_attempts:
            raise
        wait = min(60, 2 ** attempt) + random.uniform(0, 0.4)
        time.sleep(wait)
        return get_json_with_retry(session, url, params, attempt+1, max_attempts, headers)

    if r.status_code == 429:
        if attempt >= max_attempts:
            raise RuntimeError("HTTP 429 too many retries")
        ra = r.headers.get("Retry-After")
        wait = float(ra) if ra and ra.isdigit() else min(60, 2 ** attempt)
        time.sleep(wait + random.uniform(0, 0.4))
        return get_json_with_retry(session, url, params, attempt+1, max_attempts, headers)

    if r.status_code >= 500:
        if attempt >= max_attempts:
            r.raise_for_status()
        wait = min(60, 2 ** attempt) + random.uniform(0, 0.4)
        time.sleep(wait)
        return get_json_with_retry(session, url, params, attempt+1, max_attempts, headers)

    r.raise_for_status()
    return r.json()

def collect_langchain_tools(session: requests.Session, seen_keys: set):
    out = []
    resp = _get(session, LANGCHAIN_INDEX)
    soup = BeautifulSoup(resp.text, "html.parser")

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/oss/python/integrations/tools/" in href or "/oss/python/integrations/providers/" in href:
            if href.startswith("http"):
                links.add(href)
            else:
                links.add(urljoin(LANGCHAIN_BASE, href))
        elif href.startswith("/docs/integrations/tools/") or "/integrations/tools/" in href:
            links.add(urljoin(LANGCHAIN_BASE, href))
        elif href.startswith("/docs/integrations/providers/") or "/oss/python/integrations/providers/" in href:
            if "search" in href.lower() or "tool" in href.lower() or "api" in href.lower():
                if href.startswith("http"):
                    links.add(href)
                else:
                    links.add(urljoin(LANGCHAIN_BASE, href))

    for url in tqdm(sorted(links), desc="LangChain pages"):
        try:
            page = _get(session, url)
        except Exception:
            continue
        psoup = BeautifulSoup(page.text, "html.parser")
        h1 = psoup.find(["h1","h2"])
        name = norm_ws(h1.get_text()) if h1 else url.split("/")[-2]
        desc = ""
        main = psoup.find("main") or psoup
        for p in main.find_all(["p","div"], recursive=True):
            txt = norm_ws(p.get_text(" "))
            if txt and len(txt) > 40:
                desc = txt
                break

        item = {
            "source": "langchain",
            "name": name,
            "slug": to_slug(name),
            "url": url,
            "description": desc,
            "tags": ["langchain","tool-doc"],
            "raw": {
                "title": name,
                "html_len": len(page.text)
            }
        }
        key = ("langchain", item["slug"])
        if key in seen_keys: 
            continue
        seen_keys.add(key)
        out.append(item)
        time.sleep(MIN_DELAY_BETWEEN_REQUESTS)
    return out


def collect_crewai_tools(session: requests.Session, seen_keys: set, github_token: str=None):
    import base64
    headers = dict(HEADERS)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    owner = "crewAIInc"
    repo  = "crewAI-tools"
    r = session.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=30)
    r.raise_for_status()
    default_branch = r.json().get("default_branch", "main")

    r = session.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1",
        headers=headers, timeout=60
    )
    r.raise_for_status()
    tree = r.json().get("tree", [])

    prefix = "crewai_tools/tools/"
    files = [t for t in tree if t.get("type") == "blob" and t.get("path","").startswith(prefix)]

    from collections import defaultdict
    groups = defaultdict(list)
    for f in files:
        rel = f["path"][len(prefix):]
        tool_dir = rel.split("/", 1)[0]
        groups[tool_dir].append(f)

    out = []
    for tool_name, flist in sorted(groups.items()):
        readme = next((f for f in flist if f["path"].lower().endswith(("readme.md","readme.rst"))), None)
        pyfile = next((f for f in flist if f["path"].endswith(".py")), None)
        desc = ""
        if readme:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{readme['path']}"
            try:
                rr = session.get(raw_url, headers=headers, timeout=30); rr.raise_for_status()
                lines = [ln.strip() for ln in rr.text.splitlines()]
                buf = []
                for ln in lines:
                    if ln.startswith("#"):
                        if buf: break
                        continue
                    if ln:
                        buf.append(ln)
                    elif buf:
                        break
                desc = norm_ws(" ".join(buf))[:1000]
            except Exception:
                pass

        if not desc and pyfile:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{pyfile['path']}"
            try:
                pr = session.get(raw_url, headers=headers, timeout=30); pr.raise_for_status()
                m = re.search(r'"""(.*?)"""', pr.text, flags=re.S|re.M) or re.search(r"'''(.*?)'''", pr.text, flags=re.S|re.M)
                if m:
                    desc = norm_ws(m.group(1))[:1000]
            except Exception:
                pass

        item = {
            "source": "crewai",
            "name": tool_name,
            "slug": to_slug(tool_name),
            "url": f"https://github.com/{owner}/{repo}/tree/{default_branch}/{prefix}{tool_name}",
            "description": desc,
            "tags": ["crewai","tool-repo"],
            "raw": {"files": len(flist), "branch": default_branch}
        }
        key = ("crewai", item["slug"])
        if key in seen_keys: 
            continue
        seen_keys.add(key)
        out.append(item)

    return out


def collect_llamaindex_tools(session: requests.Session, seen_keys: set, github_token: str=None):
    import base64
    headers = dict(HEADERS)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    owner = LLAMAINDEX_OWNER
    repo = LLAMAINDEX_REPO
    tools_path = LLAMAINDEX_TOOLS_PATH
    r = session.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=30)
    r.raise_for_status()
    default_branch = r.json().get("default_branch", "main")
    r = session.get(
        f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1",
        headers=headers, timeout=60
    )
    r.raise_for_status()
    tree = r.json().get("tree", [])
    prefix = f"{tools_path}/"
    files = [t for t in tree if t.get("type") == "blob" and t.get("path", "").startswith(prefix)]

    from collections import defaultdict
    groups = defaultdict(list)
    for f in files:
        rel = f["path"][len(prefix):]
        tool_dir = rel.split("/", 1)[0]
        groups[tool_dir].append(f)
    out = []
    for tool_name, flist in sorted(groups.items()):
        readme = next((f for f in flist if f["path"].lower().endswith(("readme.md", "readme.rst", "readme.txt"))), None)
        pyfile = next((f for f in flist if f["path"].endswith(".py") and "__init__.py" not in f["path"]), None)
        init_file = next((f for f in flist if f["path"].endswith("__init__.py")), None)

        desc = ""
        if readme:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{readme['path']}"
            try:
                rr = session.get(raw_url, headers=headers, timeout=30)
                rr.raise_for_status()
                lines = [ln.strip() for ln in rr.text.splitlines()]
                buf = []
                for ln in lines:
                    if ln.startswith("#"):
                        if buf:
                            break
                        continue
                    if ln:
                        buf.append(ln)
                    elif buf:
                        break
                desc = norm_ws(" ".join(buf))[:1000]
            except Exception:
                pass
        if not desc and pyfile:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{pyfile['path']}"
            try:
                pr = session.get(raw_url, headers=headers, timeout=30)
                pr.raise_for_status()
                m = re.search(r'"""(.*?)"""', pr.text, flags=re.S|re.M) or re.search(r"'''(.*?)'''", pr.text, flags=re.S|re.M)
                if m:
                    desc = norm_ws(m.group(1))[:1000]
            except Exception:
                pass

        if not desc and init_file:
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{default_branch}/{init_file['path']}"
            try:
                pr = session.get(raw_url, headers=headers, timeout=30)
                pr.raise_for_status()
                m = re.search(r'"""(.*?)"""', pr.text, flags=re.S|re.M) or re.search(r"'''(.*?)'''", pr.text, flags=re.S|re.M)
                if m:
                    desc = norm_ws(m.group(1))[:1000]
            except Exception:
                pass
        if not desc:
            desc = f"LlamaIndex integration for {tool_name.replace('_', ' ').title()}"

        item = {
            "source": "llamaindex",
            "name": tool_name,
            "slug": to_slug(f"llamaindex-{tool_name}"),
            "url": f"https://github.com/{owner}/{repo}/tree/{default_branch}/{prefix}{tool_name}",
            "description": desc,
            "tags": ["llamaindex", "tool-integration", "llamahub"],
            "raw": {
                "files": len(flist),
                "branch": default_branch,
                "path": f"{prefix}{tool_name}"
            }
        }
        key = ("llamaindex", item["slug"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(item)

    return out


N8N_KEYWORDS = [
    "n8n-nodes",
    "n8n-community-node-package"
]

NODE_FILE_PATTERNS = (
    r"\.node\.(ts|js)$",
    r"Node\.ts$",
    r"node\.ts$",
)

RE_DESC_OBJECT = re.compile(
    r"description\s*:\s*{(?P<body>.*?)}\s*,", flags=re.S
)


RE_LONGDESC = re.compile(
    r"^\s*(description|subtitle|notes)\s*:\s*['\"](?P<d>[^'\"]{30,})['\"]\s*,",
    flags=re.M
)
RE_DISPLAYNAME = re.compile(r"displayName\s*:\s*['\"](?P<dn>[^'\"]+)['\"]")
RE_CREDENTIALS = re.compile(r"credentials\s*:\s*\[(?P<cre>.*?)\]", flags=re.S)
RE_CRE_NAME = re.compile(r"name\s*:\s*['\"](?P<n>[^'\"]+)['\"]")
RE_DEFAULT_NAME = re.compile(r"defaults\s*:\s*{[^}]*name\s*:\s*['\"](?P<n>[^'\"]+)['\"]", flags=re.S)

def _parse_node_ts(js_text: str):
    data = {}
    mdn = RE_DISPLAYNAME.search(js_text)
    if mdn:
        data["displayName"] = norm_ws(mdn.group("dn"))
    mdesc = RE_LONGDESC.search(js_text)
    if mdesc:
        data["longDescription"] = norm_ws(mdesc.group("d"))
    mcre = RE_CREDENTIALS.search(js_text)
    if mcre:
        names = []
        for nm in RE_CRE_NAME.finditer(mcre.group("cre")):
            names.append(nm.group("n"))
        if names:
            data["credentials"] = sorted(set(names))
    if "displayName" not in data:
        mdef = RE_DEFAULT_NAME.search(js_text)
        if mdef:
            data["displayName"] = norm_ws(mdef.group("n"))
    return data

def _download_and_parse_tarball(session: requests.Session, tar_url: str, verbose: bool=False):
    tmpdir = tempfile.mkdtemp(prefix="n8n_tar_")
    results = {}
    readme_text = ""
    pkg_json = {}

    try:
        with session.get(tar_url, headers=HEADERS, stream=True, timeout=60) as r:
            r.raise_for_status()
            tpath = os.path.join(tmpdir, "pkg.tgz")
            with open(tpath, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        with tarfile.open(tpath, "r:gz") as tarf:
            try:
                tarf.extractall(tmpdir, filter='data')
            except TypeError:
                tarf.extractall(tmpdir)
        root_candidates = []
        for entry in os.listdir(tmpdir):
            p = os.path.join(tmpdir, entry)
            if os.path.isdir(p):
                root_candidates.append(p)
        root = min(root_candidates, key=len) if root_candidates else tmpdir
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn == "package.json":
                    pj_path = os.path.join(dirpath, fn)
                    try:
                        with open(pj_path, "r", encoding="utf-8", errors="ignore") as fh:
                            pkg_json = json.load(fh)
                    except Exception:
                        pass
                    break
            if pkg_json:
                break

        if pkg_json:
            results["package"] = {
                "name": pkg_json.get("name"),
                "version": pkg_json.get("version"),
                "description": pkg_json.get("description"),
                "keywords": pkg_json.get("keywords"),
                "repository": (pkg_json.get("repository") or {}).get("url") if isinstance(pkg_json.get("repository"), dict) else pkg_json.get("repository"),
                "homepage": pkg_json.get("homepage"),
                "author": pkg_json.get("author"),
                "license": pkg_json.get("license"),
                "engines": pkg_json.get("engines"),
                "n8n": pkg_json.get("n8n")
            }
        for dirpath, _, files in os.walk(root):
            for fn in files:
                lower = fn.lower()
                if lower.startswith("readme") and lower.endswith((".md", ".rst", ".txt")):
                    try:
                        with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as fh:
                            readme_text = fh.read()
                    except Exception:
                        pass
                    break
            if readme_text:
                break

        node_infos = []
        for dirpath, _, files in os.walk(root):
            for fn in files:
                full = os.path.join(dirpath, fn)
                low = full.lower()
                if any(low.endswith(suffix) for suffix in (".node.ts", ".node.js", "node.ts", "node.js")) or low.endswith("node.ts"):
                    try:
                        with open(full, "r", encoding="utf-8", errors="ignore") as fh:
                            txt = fh.read()
                        meta = _parse_node_ts(txt)
                        if meta:
                            node_infos.append(meta)
                    except Exception:
                        pass
        if node_infos:
            dn = next((x.get("displayName") for x in node_infos if x.get("displayName")), None)
            desc = next((x.get("longDescription") for x in node_infos if x.get("longDescription")), None)
            creds = sorted(set(sum([x.get("credentials", []) for x in node_infos if x.get("credentials")], [])))
            if dn:
                results["displayName"] = dn
            if desc:
                results["longDescription"] = desc
            if creds:
                results["credentials"] = creds
        if readme_text and not results.get("longDescription"):
            lines = [ln.strip() for ln in readme_text.splitlines()]
            buf = []
            for ln in lines:
                if ln.startswith("#"):
                    if buf: break
                    continue
                if ln:
                    buf.append(ln)
                elif buf:
                    break
            if buf:
                results["longDescription"] = norm_ws(" ".join(buf))[:1000]

        if verbose:
            print(f"    [tar] package.json={'yes' if pkg_json else 'no'}  "
                  f"node_files={len(node_infos)}  readme={'yes' if readme_text else 'no'}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return results

def load_n8n_from_csv(csv_path: Path, seen_keys: set, verbose: bool=False):
    out = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        script_dir = Path(__file__).parent.parent.parent
        alt_path = script_dir / csv_path
        if alt_path.exists():
            csv_path = alt_path
        else:
            cwd_path = Path.cwd() / csv_path
            if cwd_path.exists():
                csv_path = cwd_path
            else:
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}\n"
                    f"  Tried: {csv_path}\n"
                    f"  Tried: {alt_path}\n"
                    f"  Tried: {cwd_path}\n"
                    f"  Current working directory: {Path.cwd()}"
                )
    print(f"[n8n] Loading from CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig") as f:
        total_rows = sum(1 for _ in f) - 1
    with csv_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=total_rows, desc="  n8n CSV load", unit="row", leave=True)
        
        for idx, row in enumerate(reader, 1):
            name = row.get("name", "").strip()
            if not name:
                pbar.update(1)
                continue
            item = {
                "source": "n8n",
                "name": name,
                "slug": to_slug(name),
                "url": row.get("npm_url", "").strip() or f"https://www.npmjs.com/package/{name}",
                "description": norm_ws(row.get("description", "")),
                "tags": ["n8n", "community-node"],
                "raw": {
                    "version": row.get("version", "").strip(),
                    "date": row.get("date", "").strip(),
                    "homepage": normalize_repo(row.get("homepage_url", "").strip()),
                    "repository": normalize_repo(row.get("repository_url", "").strip()),
                    "weekly_downloads": int(row.get("weekly_downloads", 0) or 0),
                }
            }
            
            key = ("n8n", item["slug"])
            if key in seen_keys:
                pbar.update(1)
                continue
            seen_keys.add(key)
            out.append(item)
            
            pbar.update(1)
            if idx % 100 == 0:
                pbar.set_postfix({"kept": len(out)})
        pbar.close()
    print(f"[n8n] CSV load done: kept {len(out)} items")
    return out

def collect_n8n_nodes(session: requests.Session, seen_keys: set,
                      pages: int=0, verbose: bool=False,
                      query_mode: str="name", text: str="n8n-nodes",
                      fetch_all: bool=True, hydrate_missing: bool=True):
    out = []
    size = 250
    text_lower = text.lower()
    scanned, kept = 0, 0
    if query_mode == "name":
        queries = [text]
    else:
        queries = [f"keywords:{text}", "keywords:n8n-community-node-package"]
    if pages == 0:
        fetch_all = True
        if verbose:
            print("[n8n] pages=0, fetch_all enabled (crawl all results)")
    for q in queries:
        print(f"[n8n] Search query: {q}")
        if fetch_all:
            print("  -> Will fetch all results (no page limit)")
        elif pages > 0:
            print(f"  -> Up to {pages} pages (size={size} per page)")
        
        offset = 0
        page_idx = 0
        total = None
        pbar = None
        
        while True:
            if (not fetch_all) and (pages > 0) and (page_idx >= pages):
                if verbose:
                    print(f"  (page limit {pages} reached, stopping)")
                break
            try:
                data = get_json_with_retry(session, NPM_SEARCH, 
                                         params={"text": q, "size": size, "from": offset},
                                         headers={"User-Agent": "n8n-node-research/2.0", "Accept": "application/json"})
            except Exception as e:
                if verbose:
                    print(f"  [ERR] Search failed: {e}")
                break
            objects = data.get("objects", []) or []
            total = data.get("total", total)
            page_idx += 1
            scanned += len(objects)
            if pbar is None and total is not None:
                pbar = tqdm(total=total, desc="  n8n collect", unit="pkg", leave=True)
                pbar.update(offset)
            if pbar is not None:
                pbar.update(len(objects))
                pbar.set_postfix({"kept": kept, "page": page_idx})
            if verbose:
                print(f"  [page {page_idx}] returned {len(objects)} | total {total}")
            if not objects:
                if verbose:
                    print("  (no more results)")
                break

            page_kept = 0
            page_skipped = 0

            for idx, obj in enumerate(objects, 1):
                pkg = obj.get("package", {}) or {}
                name = pkg.get("name") or ""
                if query_mode == "name" and text_lower not in name.lower():
                    page_skipped += 1
                    continue
                try:
                    preg = get_json_with_retry(session, f"{NPM_PKG_BASE}/{safe_pkg(name)}")
                except Exception as e:
                    if verbose:
                        print(f"    [WARN] {name}: registry fetch failed: {e}")
                    continue
                latest = safe_get(preg, "dist-tags", "latest")
                vermeta = safe_get(preg, "versions", latest) or {}
                tarball = safe_get(vermeta, "dist", "tarball")
                repo_url = normalize_repo(pkg.get("links", {}).get("repository") or safe_get(vermeta, "repository", "url") or "")
                home_url = pkg.get("links", {}).get("homepage") or vermeta.get("homepage") or ""
                if hydrate_missing and (not repo_url or not home_url):
                    r2, h2 = get_repo_home_from_registry(session, name)
                    if not repo_url and r2:
                        repo_url = r2
                        if verbose:
                            print(f"    [hydrate] {name}: repository = {repo_url}")
                    if not home_url and h2:
                        home_url = h2
                        if verbose:
                            print(f"    [hydrate] {name}: homepage = {home_url}")

                item = {
                    "source": "n8n",
                    "name": name,
                    "slug": to_slug(name),
                    "url": pkg.get("links", {}).get("npm") or f"https://www.npmjs.com/package/{name}",
                    "description": norm_ws(pkg.get("description") or vermeta.get("description") or ""),
                    "tags": sorted(set((pkg.get("keywords") or []) + ["n8n","community-node"])),
                    "raw": {
                        "version": latest,
                        "homepage": home_url,
                        "repository": repo_url,
                        "maintainers": [m.get("username") for m in (pkg.get("maintainers") or []) if isinstance(m, dict)],
                    }
                }
                weekly_downloads = get_weekly_downloads(session, name)
                item["raw"]["weekly_downloads"] = weekly_downloads

                if verbose:
                    print(f"  [{page_idx}:{idx:03d}] {name}  v={latest}  tarball={'yes' if tarball else 'no'}  downloads={weekly_downloads}")
                elif pbar is not None:
                    if kept % 10 == 0:
                        pbar.set_description(f"  n8n collect (kept {kept})")
                if tarball:
                    try:
                        extra = _download_and_parse_tarball(session, tarball, verbose=verbose)
                        if "package" in extra:
                            pj = extra["package"]
                            item["raw"]["package"] = pj
                            if (not item["description"]) and pj.get("description"):
                                item["description"] = norm_ws(pj["description"])
                            if pj.get("keywords"):
                                item["tags"] = sorted(set(item["tags"] + pj["keywords"]))
                            if pj.get("repository") and not item["raw"].get("repository"):
                                item["raw"]["repository"] = normalize_repo(pj.get("repository") or "")
                            if pj.get("homepage") and not item["raw"].get("homepage"):
                                item["raw"]["homepage"] = pj["homepage"]
                            for k in ("license","author","engines","n8n"):
                                if pj.get(k): item["raw"][k] = pj[k]
                        if extra.get("displayName"):
                            item["display_name"] = extra["displayName"]
                        if extra.get("longDescription") and (not item["description"] or len(item["description"]) < 40):
                            item["description"] = extra["longDescription"]
                        if extra.get("credentials"):
                            item["credentials"] = extra["credentials"]
                    except Exception as e:
                        if verbose:
                            print(f"    [tar] parse error: {e}")

                key = ("n8n", item["slug"])
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                out.append(item)
                page_kept += 1
                kept += 1

            if verbose:
                print(f"  [page {page_idx} done] wrote {page_kept}, skipped {page_skipped} (non-match)")
            offset += len(objects)
            if fetch_all:
                if total is not None and offset >= total:
                    if verbose:
                        print(f"  (reached API total: {total})")
                    break
            else:
                if len(objects) < size:
                    if verbose:
                        print("  (last page, done)")
                    break
            sleep_s = max(MIN_DELAY_BETWEEN_REQUESTS, 0.8 + random.uniform(0, 0.3))
            if verbose and page_idx % 5 == 0:
                print(f"  [sleep] {sleep_s:.1f}s ...")
            time.sleep(sleep_s)
        if pbar is not None:
            pbar.close()
    print(f"[n8n] Done: scanned {scanned}, kept {kept}")
    if verbose:
        print(f"  Scanned={scanned}, Kept={kept}, Total={total}")
    return out


def collect_n8n_official_github(session: requests.Session, seen_keys: set, github_token: str=None, verbose: bool=False):
    headers = dict(HEADERS)
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    r = session.get(f"https://api.github.com/repos/{N8N_OFFICIAL_OWNER}/{N8N_OFFICIAL_REPO}",
                    headers=headers, timeout=30)
    r.raise_for_status()
    default_branch = r.json().get("default_branch", "master")
    r = session.get(
        f"https://api.github.com/repos/{N8N_OFFICIAL_OWNER}/{N8N_OFFICIAL_REPO}"
        f"/git/trees/{default_branch}?recursive=1",
        headers=headers, timeout=60
    )
    r.raise_for_status()
    tree = r.json().get("tree", [])

    prefix = f"{N8N_OFFICIAL_PATH}/"
    files = [t for t in tree if t.get("type") == "blob" and t.get("path","").startswith(prefix)]
    from collections import defaultdict
    groups = defaultdict(list)
    for f in files:
        rel = f["path"][len(prefix):]
        top = rel.split("/", 1)[0]
        groups[top].append(f)
    out = []
    for node_dir, flist in sorted(groups.items()):
        readme = next((f for f in flist if f["path"].lower().endswith(("readme.md","readme.rst","readme.txt"))), None)
        node_files = [f for f in flist if f["path"].lower().endswith((".node.ts",".node.js","node.ts","node.js"))]

        display_name = None
        long_desc = ""
        creds = []
        for nf in node_files:
            raw_url = f"https://raw.githubusercontent.com/{N8N_OFFICIAL_OWNER}/{N8N_OFFICIAL_REPO}/{default_branch}/{nf['path']}"
            try:
                r2 = session.get(raw_url, headers=headers, timeout=30); r2.raise_for_status()
                txt = r2.text
                meta = _parse_node_ts(txt)
                if meta.get("displayName") and not display_name:
                    display_name = meta["displayName"]
                if meta.get("longDescription") and len(long_desc) < len(meta["longDescription"]):
                    long_desc = meta["longDescription"]
                if meta.get("credentials"):
                    creds = sorted(set(creds + meta["credentials"]))
            except Exception as e:
                if verbose:
                    print(f"    [n8n-official] parse node file error: {e}")
        if not long_desc and readme:
            raw_url = f"https://raw.githubusercontent.com/{N8N_OFFICIAL_OWNER}/{N8N_OFFICIAL_REPO}/{default_branch}/{readme['path']}"
            try:
                rr = session.get(raw_url, headers=headers, timeout=30); rr.raise_for_status()
                lines = [ln.strip() for ln in rr.text.splitlines()]
                buf = []
                for ln in lines:
                    if ln.startswith("#"):
                        if buf: break
                        continue
                    if ln:
                        buf.append(ln)
                    elif buf:
                        break
                if buf:
                    long_desc = norm_ws(" ".join(buf))[:1000]
            except Exception as e:
                if verbose:
                    print(f"    [n8n-official] read README error: {e}")

        name = display_name or node_dir
        item = {
            "source": "n8n_official",
            "name": name,
            "slug": to_slug(f"n8n-official-{node_dir}"),
            "url": f"https://github.com/{N8N_OFFICIAL_OWNER}/{N8N_OFFICIAL_REPO}/tree/{default_branch}/{prefix}{node_dir}",
            "description": long_desc,
            "tags": ["n8n","official-node","nodes-base"],
            "credentials": creds or None,
            "raw": {
                "branch": default_branch,
                "files": len(flist)
            }
        }
        key = ("n8n_official", item["slug"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(item)

        if verbose:
            print(f"  [n8n-official] + {node_dir}  display={bool(display_name)} readme={bool(readme)} creds={len(creds)}")

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="Print verbose progress for n8n crawling")

    ap.add_argument("--out", type=str, default=f"tools_{EXPORT_TS.strftime('%Y%m%dT%H%M%SZ')}.jsonl", help="Output JSONL path")
    ap.add_argument("--sources", nargs="+", default=["langchain","crewai","n8n","n8n_official"],
                choices=["langchain","crewai","n8n","n8n_official","llamaindex"])

    ap.add_argument("--github-token", type=str, default=None,
                help="GitHub token for API rate limits (default: $GITHUB_TOKEN env var, or None if not set)")
    ap.add_argument("--npm-pages", type=int, default=0, help="Pages per query (size=250). 0=fetch all (default), >0=limit pages")
    ap.add_argument("--n8n-query-mode", choices=["keywords","name"], default="name",
                help="n8n search: keywords=by keyword; name=by package name substring (broader)")
    ap.add_argument("--n8n-text", default="n8n-nodes",
                help="Search text when --n8n-query-mode=name (default n8n-nodes)")
    ap.add_argument("--no-npm-fetch-all", dest="npm_fetch_all", action="store_false", default=True,
                help="Disable fetch_all; use --npm-pages to limit (default: fetch_all on)")
    ap.add_argument("--hydrate-missing", action="store_true", default=True,
                help="Fill missing repository/homepage from registry (default on)")
    ap.add_argument("--n8n-csv", type=str, default=None,
                help="If set, load n8n from CSV instead of crawling (e.g. data/n8n_nodes_final_2025-11-07_10-52-09.csv)")
    args = ap.parse_args()
    if args.github_token is None:
        args.github_token = os.environ.get("GITHUB_TOKEN")

    outp = Path(args.out)
    if outp.exists():
        print(f"[!] Append mode: {outp}")
    seen = set()
    session = requests.Session()
    if args.sources:
        print("[Note] Using polite delays; research use only. Comply with each source's terms of service.\n")

    total = 0

    if "langchain" in args.sources:
        print("[LangChain] crawling docs…")
        lc = collect_langchain_tools(session, seen)
        write_jsonl(outp, lc)
        print(f"  -> {len(lc)} items")
        total += len(lc)

    if "crewai" in args.sources:
        print("[CrewAI] listing repo…")
        cr = collect_crewai_tools(session, seen, args.github_token)
        write_jsonl(outp, cr)
        print(f"  -> {len(cr)} items")
        total += len(cr)

    if "llamaindex" in args.sources:
        print("[LlamaIndex] listing tools integration repo…")
        li = collect_llamaindex_tools(session, seen, args.github_token)
        write_jsonl(outp, li)
        print(f"  -> {len(li)} items")
        total += len(li)

    if "n8n" in args.sources:
        if args.n8n_csv:
            csv_path = Path(args.n8n_csv)
            if not csv_path.is_absolute() and not csv_path.exists():
                project_root = Path(__file__).parent.parent.parent
                alt_path = project_root / csv_path
                if alt_path.exists():
                    csv_path = alt_path
            nn = load_n8n_from_csv(csv_path, seen, verbose=args.verbose)
        else:
            print("[n8n] npm registry search & tarball parse…")
            npm_pages = args.npm_pages
            fetch_all_setting = args.npm_fetch_all if npm_pages > 0 else True
            nn = collect_n8n_nodes(
                session, seen,
                pages=npm_pages,
                verbose=args.verbose,
                query_mode=args.n8n_query_mode,
                text=args.n8n_text,
                fetch_all=fetch_all_setting,
                hydrate_missing=args.hydrate_missing
            )

        write_jsonl(outp, nn)
        print(f"  -> {len(nn)} items")
        total += len(nn)

    if "n8n_official" in args.sources:
        print("[n8n-official] GitHub nodes-base crawling…")
        noff = collect_n8n_official_github(session, seen_keys=seen, github_token=args.github_token, verbose=args.verbose)
        write_jsonl(outp, noff)
        print(f"  -> {len(noff)} items")
        total += len(noff)

    print(f"\n[Done] total items appended: {total}\nOutput: {outp.resolve()}")

if __name__ == "__main__":
    main()
