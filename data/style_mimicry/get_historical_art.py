#!/usr/bin/env python3
"""Fetch historical artist images from WikiArt until target sample counts are met."""

from __future__ import annotations

import argparse, sys, time, os
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import requests
from PIL import Image, UnidentifiedImageError

try:
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException, WebDriverException
    from selenium.webdriver.common.by import By
except ModuleNotFoundError:
    webdriver = None  # type: ignore[assignment]
    NoSuchElementException = Exception  # type: ignore[assignment]
    WebDriverException = Exception  # type: ignore[assignment]
    By = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICRY_DIR = REPO_ROOT / "mimicry"
if str(MIMICRY_DIR) not in sys.path:
    sys.path.insert(0, str(MIMICRY_DIR))

from common.progress_tracker import ProgressTracker  # noqa: E402


WIKIART_URL_PREFIX = "https://www.wikiart.org/en/"
WIKIART_URL_SUFFIX = "/all-works#!#filterName:Style_"
WIKIART_URL_SUFFIX_2 = ",resultType:masonry"
MASONRY_SELECTOR = "div.masonry-content"
LOAD_MORE_SELECTOR = (
    "body > div.wiki-container > div.wiki-container-responsive.with-overflow > "
    "section > main > div.view-all-works.ng-scope > div.masonry-outter-container > "
    "div:nth-child(2) > a"
)
PLACEHOLDER_TAG = "lazy-load-placeholder"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fill historical artist folders to target image counts.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./"),
        help="Dataset root path (default: data/style_mimicry)",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=20,
        help="Required images per artist",
    )
    parser.add_argument(
        "--max-load-more-clicks",
        type=int,
        default=2,
        help="Maximum Load More clicks per artist page",
    )
    parser.add_argument(
        "--max-scrolls",
        type=int,
        default=5,
        help="Maximum lazy-load scroll operations per artist page",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=20,
        help="HTTP timeout in seconds for image downloads",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=.5,
        help="Delay between browser actions to stabilize lazy loading",
    )
    parser.add_argument(
        "--artist",
        action="append",
        default=[],
        help="Optional artist directory name(s) to process (repeatable)",
    )
    parser.add_argument(
        "--style",
        action="append",
        default=[],
        help="Optional style directory name(s) to process (repeatable)",
    )
    parser.add_argument(
        "--limit-artists",
        type=int,
        default=0,
        help="Process at most N artists (0 means all selected artists)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run without making web requests",
    )
    return parser.parse_args()


def sanitize_filename(text: str, fallback: str) -> str:
    """Sanitize title/alt text into a filesystem-safe filename."""
    cleaned = "".join(char for char in text if char.isalnum() or char in (" ", "_", "-")).strip()
    return cleaned or fallback


def existing_images(directory: Path) -> int:
    """Count existing image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    count = 0
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in {".jpg", ".jpeg", ".png"}:
            count += 1
    return count


def dedup_key(name: str) -> str:
    """Extract lowercase alphanumeric characters from a name for duplicate comparison."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def remove_near_duplicate_filenames(directory: Path) -> int:
    """Remove duplicate filenames in a directory, keeping the first of each group.

    Two filenames are considered duplicates when their basenames (without
    extension) contain the same alphanumeric characters in the same order,
    ignoring case, spaces, and punctuation.  Returns the number of files removed.
    """
    if not os.path.isdir(directory):
        return 0
    files = sorted(
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png"}
    )
    seen_keys: dict[str, str] = {}
    removed = 0
    for f in files:
        key = dedup_key(os.path.splitext(f)[0])
        if key in seen_keys:
            path = os.path.join(directory, f)
            os.remove(path)
            print(f"  removed duplicate: {f} (kept {seen_keys[key]})")
            removed += 1
        else:
            seen_keys[key] = f
    return removed


def build_artist_search_url(artist_slug: str, style: str) -> str:
    """Build the WikiArt search URL for one artist slug."""
    return f"{WIKIART_URL_PREFIX}{quote_plus(artist_slug)}{WIKIART_URL_SUFFIX}{quote_plus(style)}{WIKIART_URL_SUFFIX_2}"


def new_driver() -> webdriver.Chrome:
    """Create a headless Chrome webdriver instance."""
    if webdriver is None:
        raise ModuleNotFoundError("selenium is required for non-dry-run scraping.")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=options)


def collect_image_candidates(
    driver: webdriver.Chrome,
    artist_slug: str,
    art_style: str,
    target_count: int,
    max_load_more_clicks: int,
    max_scrolls: int,
    sleep_seconds: float,
) -> list[tuple[str, str]]:
    """Collect candidate (url, alt) image tuples from WikiArt masonry content."""
    
    search_url = build_artist_search_url(artist_slug=artist_slug, style=art_style)
    if By is None:
        raise ModuleNotFoundError("selenium is required for non-dry-run scraping.")
    driver.get(search_url)
    time.sleep(sleep_seconds)

    try:
        load_more = driver.find_element(By.CSS_SELECTOR, LOAD_MORE_SELECTOR)
    except NoSuchElementException:
        load_more = None

    for _ in range(max_load_more_clicks):
        if load_more is None:
            break
        try:
            driver.execute_script("arguments[0].click();", load_more)
            time.sleep(sleep_seconds)
        except WebDriverException:
            break

    for _ in range(max_scrolls):
        driver.execute_script("window.scrollBy(0, 900);")
        time.sleep(0.2)
        imgs = driver.find_elements(By.CSS_SELECTOR, "div.masonry-content img")
        real_count = sum(1 for image in imgs if PLACEHOLDER_TAG not in (image.get_attribute("src") or ""))
        if real_count >= target_count:
            break

    masonry = driver.find_element(By.CSS_SELECTOR, MASONRY_SELECTOR)
    image_elements = masonry.find_elements(By.TAG_NAME, "img")
    candidates: list[tuple[str, str]] = []
    seen_urls: set[str] = set()
    for image in image_elements:
        src = image.get_attribute("data-src") or image.get_attribute("src") or ""
        if not src or PLACEHOLDER_TAG in src:
            continue
        clean_src = src.split("!")[0]
        if clean_src in seen_urls:
            continue
        seen_urls.add(clean_src)
        candidates.append((clean_src, image.get_attribute("alt") or "untitled"))
        if len(candidates) >= (target_count * 3):
            break
    return candidates


def download_candidates(
    candidates: Iterable[tuple[str, str]],
    save_dir: Path,
    needed_count: int,
    request_timeout: int,
) -> int:
    """Download candidate URLs until needed_count images are saved."""
    saved = 0
    used_names: set[str] = {os.path.splitext(path)[0] for path in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, path))}
    used_keys: set[str] = {dedup_key(name) for name in used_names}

    for index, (url, alt_text) in enumerate(candidates, start=1):
        if saved >= needed_count:
            break
        base_name = sanitize_filename(text=alt_text, fallback=f"sample_{index}")
        candidate_key = dedup_key(base_name)
        if candidate_key in used_keys:
            print(f"  skipped duplicate candidate: {alt_text}")
            continue
        file_name = base_name
        suffix = 1
        while file_name in used_names:
            file_name = f"{base_name}_{suffix}"
            suffix += 1
        destination = os.path.join(save_dir, f"{file_name}.jpg")

        try:
            response = requests.get(url, timeout=request_timeout)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image.save(destination, format="JPEG", quality=95)
            used_names.add(file_name)
            used_keys.add(candidate_key)
            saved += 1
            print(f"  saved {saved}/{needed_count}: {destination}")
        except (requests.RequestException, UnidentifiedImageError, OSError) as error:
            print(f"  skipped candidate due to download/parse error: {error}")
            continue
    return saved


def list_target_artists(data_root: Path, styles: set[str], artists: set[str]) -> list[tuple[str, str, Path]]:
    """Return a sorted list of (style, artist, artist_dir) to process."""
    historical_root = os.path.join(data_root, "original_art", "historical")
    targets: list[tuple[str, str, Path]] = []

    for style_dir in sorted(path for path in os.listdir(historical_root) if os.path.isdir(os.path.join(historical_root, path))):
        style_name = style_dir
        if styles and style_name not in styles:
            continue
        for artist_dir in sorted(path for path in os.listdir(os.path.join(historical_root, style_dir)) if os.path.isdir(os.path.join(historical_root, style_dir, path))):
            artist_name = artist_dir
            if artists and artist_name not in artists:
                continue
            targets.append((style_name, artist_name, os.path.join(historical_root, style_name, artist_dir)))
    return targets


def main() -> int:
    """Execute scrape/fill flow and record per-artist bookmarks."""
    args = parse_args()
    style_filter = set(args.style)
    artist_filter = set(args.artist)
    targets = list_target_artists(data_root=args.data_root, styles=style_filter, artists=artist_filter)
    if args.limit_artists > 0:
        targets = targets[: args.limit_artists]

    tracker = ProgressTracker(
        tracker_path=args.data_root / "progress" / "progress_tracker.json",
        data_root=args.data_root,
        min_samples_per_artist=args.target_count,
    )
    tracker.load()

    if not targets:
        print("No matching artist directories were found.")
        tracker.refresh_snapshot()
        return 0

    print(f"Processing {len(targets)} artist directories with target count {args.target_count}.")
    if args.dry_run:
        for style_name, artist_name, artist_dir in targets:
            current = existing_images(artist_dir)
            print(f"[dry-run] {style_name}/{artist_name}: current={current}, needed={max(args.target_count - current, 0)}")
        tracker.refresh_snapshot()
        return 0

    driver: webdriver.Chrome | None = None
    try:
        driver = new_driver()
        for style_name, artist_name, artist_dir in targets:
            bookmark_key = f"{style_name}/{artist_name}"

            # Deduplicate existing files before counting or downloading.
            removed = remove_near_duplicate_filenames(artist_dir)
            if removed:
                print(f"\n[{bookmark_key}] removed {removed} near-duplicate(s) from existing files")

            current_count = existing_images(artist_dir)
            needed = max(args.target_count - current_count, 0)

            print(f"\n[{bookmark_key}] current={current_count}, needed={needed}")
            if needed == 0:
                tracker.set_bookmark(
                    step_name="step_1_collect_original_art",
                    bookmark_key=bookmark_key,
                    payload={
                        "status": "already_complete",
                        "style": style_name,
                        "artist": artist_name,
                        "image_count": current_count,
                    },
                )
                continue

            artist_slug = artist_name.lower().replace("_", "-")
            try:
                candidates = collect_image_candidates(
                    driver=driver,
                    artist_slug=artist_slug,
                    art_style=style_name,
                    target_count=args.target_count,
                    max_load_more_clicks=args.max_load_more_clicks,
                    max_scrolls=args.max_scrolls,
                    sleep_seconds=args.sleep_seconds,
                )
                saved_now = download_candidates(
                    candidates=candidates,
                    save_dir=artist_dir,
                    needed_count=needed,
                    request_timeout=args.request_timeout,
                )
                final_count = existing_images(artist_dir)
                status = "complete" if final_count >= args.target_count else "incomplete"
                tracker.set_bookmark(
                    step_name="step_1_collect_original_art",
                    bookmark_key=bookmark_key,
                    payload={
                        "status": status,
                        "style": style_name,
                        "artist": artist_name,
                        "saved_this_run": saved_now,
                        "image_count": final_count,
                        "target_count": args.target_count,
                    },
                )
                print(f"[{bookmark_key}] final={final_count}, status={status}")
            except (NoSuchElementException, WebDriverException, requests.RequestException) as error:
                tracker.set_bookmark(
                    step_name="step_1_collect_original_art",
                    bookmark_key=bookmark_key,
                    payload={
                        "status": "failed",
                        "style": style_name,
                        "artist": artist_name,
                        "image_count": current_count,
                        "error": str(error),
                    },
                )
                print(f"[{bookmark_key}] failed: {error}")
    finally:
        if driver is not None:
            driver.quit()

    tracker.refresh_snapshot()
    print(f"\nUpdated tracker: {(args.data_root / 'progress' / 'progress_tracker.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
