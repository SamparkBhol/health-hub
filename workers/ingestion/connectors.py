from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from html.parser import HTMLParser

from .diseases import DiseaseLexicon
from .idsp import IdspCatalogueRow, IdspConnector, parse_idsp_catalogue_text
from .language import route_unicode
from .models import Document, ExtractedSignal, FetchReceipt, LanguageRoute
from .parse import OcrHook, parse_document, tesseract_pdf_ocr
from .pipeline import IngestionPipeline
from .registry import SourceRegistry, SourceSpec
from .safe_fetch import FetchPolicy, fetch_url
from .urls import canonicalize_discovered_url


@dataclass(frozen=True, slots=True)
class DiscoveredLink:
    url: str
    label: str
    content_hint: str
    context: str = ""
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    receipt: FetchReceipt
    signal: ExtractedSignal | None
    discovered_links: tuple[DiscoveredLink, ...] = ()
    catalogue_rows: tuple[IdspCatalogueRow, ...] = ()
    processing_state: str = "parsed_redacted_evidence"


# PDFs are the only fetched bytes handed to an external renderer, so they keep a
# sandbox even though a human no longer pre-approves each digest.  The bound is
# on *size*, not identity: anything at or below this is parsed inside the
# existing structure validator (page count, per-page and total raster pixel
# ceilings) and the lease-bounded OCR budget.  Anything larger still needs a
# deliberate digest promotion, because a 20 MB scan is where render cost, not
# provenance, becomes the risk.
AUTOMATIC_PDF_PARSE_BYTE_LIMIT = 8 * 1024 * 1024


# Health vocabulary used to rank discovered links.  It is deliberately broader
# than the disease lexicon: an index row reading "ବିଜ୍ଞପ୍ତି — ସ୍ୱାସ୍ଥ୍ୟ ବିଭାଗ" is a
# health notice even when it names no disease.
HEALTH_MARKERS: tuple[str, ...] = (
    "health",
    "disease",
    "outbreak",
    "epidemic",
    "surveillance",
    "hospital",
    "medical",
    "vector borne",
    "vector-borne",
    "immunisation",
    "immunization",
    "vaccination",
    "public health",
    "sanitation",
    "drinking water",
    "advisory",
    "स्वास्थ्य",
    "रोग",
    "बीमारी",
    "प्रकोप",
    "अस्पताल",
    "टीकाकरण",
    "चिकित्सा",
    "ସ୍ୱାସ୍ଥ୍ୟ",
    "ରୋଗ",
    "ପ୍ରକୋପ",
    "ଡାକ୍ତରଖାନା",
    "ଚିକିତ୍ସା",
    "ଟୀକାକରଣ",
)

NOTICE_MARKERS: tuple[str, ...] = (
    "notification",
    "notice",
    "circular",
    "bulletin",
    "order",
    "guideline",
    "report",
    "press release",
    "situation",
    "अधिसूचना",
    "परिपत्र",
    "दिशानिर्देश",
    "ବିଜ୍ଞପ୍ତି",
    "ପରିପତ୍ର",
    "ନିର୍ଦ୍ଦେଶିକା",
)

# Never worth a detail fetch: accessibility, session and syndication surfaces.
CHROME_PATH_MARKERS: tuple[str, ...] = (
    "screen-reader",
    "screen_reader",
    "sitemap",
    "site-map",
    "site_map",
    "wp-login",
    "login",
    "logout",
    "register",
    "feedback",
    "rss",
    "feed",
    "photogallery",
    "photo-gallery",
    "video-gallery",
    "brokenlinks",
    "help",
    "font=",
    "theme=",
    "background=",
)

_NAVIGATION_SEGMENTS = frozenset(
    {
        "author",
        "authors",
        "category",
        "categories",
        "tag",
        "tags",
        "topic",
        "topics",
        "search",
    }
)
_SECTION_ONLY_PATHS = frozenset(
    {
        "crime",
        "district",
        "health",
        "lifestyle",
        "news",
        "odisha",
        "photo",
        "photos",
        "politics",
        "sports",
        "video",
        "videos",
    }
)


def is_navigation_url(url: str) -> bool:
    """Identify index/chrome URLs that must never become evidence documents."""

    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path).strip("/").casefold()
    segments = tuple(segment for segment in path.split("/") if segment)
    if not segments:
        return True
    if any(segment in _NAVIGATION_SEGMENTS for segment in segments):
        return True
    if len(segments) == 1 and segments[0] in _SECTION_ONLY_PATHS:
        return True
    if len(segments) >= 2 and segments[-2] == "page" and segments[-1].isdigit():
        return True
    return False

# Anchor text that describes the control, not the document behind it.
_GENERIC_LABEL = re.compile(
    r"^(?:download|view|details?|click here|here|read more|more|pdf|open|link|next|previous"
    r"|ଦୃଶ୍ୟ|ଡାଉନଲୋଡ|ଅଧିକ|देखें|डाउनलोड|अधिक|\d+|[\s.,:;|/()\[\]-]*)"
    r"(?:\s*\(.*\))?$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _Frame:
    tag: str
    chunks: list[str]


class _LinkParser(HTMLParser):
    """Collect anchors together with the text of the row that carries them."""

    _ROW_TAGS = frozenset({"tr", "li", "article", "section", "dd"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str, _Frame | None]] = []
        self._href: str | None = None
        self._label: list[str] = []
        self._frames: list[_Frame] = []
        self._anchor_frame: _Frame | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = tag.casefold()
        if lowered in self._ROW_TAGS:
            self._frames.append(_Frame(tag=lowered, chunks=[]))
            return
        if lowered != "a":
            return
        values = {str(key).casefold(): str(value) for key, value in attrs if value is not None}
        self._href = values.get("href")
        self._label = []
        self._anchor_frame = self._frames[-1] if self._frames else None

    def handle_data(self, data: str) -> None:
        if self._frames:
            self._frames[-1].chunks.append(data)
        if self._href is not None:
            self._label.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "a" and self._href is not None:
            self.links.append(
                (self._href, " ".join("".join(self._label).split()), self._anchor_frame)
            )
            self._href = None
            self._label = []
            self._anchor_frame = None
            return
        if lowered in self._ROW_TAGS:
            for index in range(len(self._frames) - 1, -1, -1):
                if self._frames[index].tag == lowered:
                    del self._frames[index:]
                    return


def _frame_text(frame: _Frame | None) -> str:
    if frame is None:
        return ""
    return " ".join("".join(frame.chunks).split())


def _is_generic_label(label: str) -> bool:
    return bool(_GENERIC_LABEL.match(label.strip()))


def score_link(
    *,
    url: str,
    label: str,
    context: str,
    content_hint: str,
    source: SourceSpec,
    lexicon: DiseaseLexicon | None = None,
) -> float:
    """Score a discovered link by how likely it is to carry health evidence."""

    haystack = f"{label} {context}".casefold()
    path = url.casefold()
    score = 0.0
    if lexicon is not None and lexicon.find(haystack):
        score += 5.0
    if any(marker in haystack for marker in HEALTH_MARKERS):
        score += 3.0
    if any(marker in haystack for marker in NOTICE_MARKERS):
        score += 1.0
    if content_hint == "application/pdf":
        score += 2.0 if "pdf" in source.kind.casefold() else 1.0
    if any(marker in path for marker in CHROME_PATH_MARKERS):
        score -= 4.0
    if is_navigation_url(url):
        score -= 10.0
    if not label.strip():
        score -= 1.0
    return score


def discover_registered_links(
    body: bytes,
    *,
    index_url: str,
    source: SourceSpec,
    maximum_links: int = 200,
    lexicon: DiseaseLexicon | None = None,
) -> tuple[DiscoveredLink, ...]:
    """Extract same-host detail and PDF links from an index page, best first.

    Index pages are navigation chrome wrapped around a list of notices. The
    anchor of a notice is often just "Download(122.6 KB)", so the row text that
    carries it is kept as context and folded into the label; ranking then puts
    the health rows ahead of the site furniture. Host allowlisting, the link
    cap and the caller's fetch budget are unchanged.
    """

    parser = _LinkParser()
    parser.feed(body.decode("utf-8", errors="replace"))
    allowed_hosts = set(source.allowed_hosts)
    found: dict[str, DiscoveredLink] = {}
    for raw_url, anchor_text, frame in parser.links:
        absolute = urllib.parse.urljoin(index_url, raw_url)
        parsed = urllib.parse.urlsplit(absolute)
        host = (parsed.hostname or "").lower()
        # The fetch policy permits HTTPS only, so an `http://` row on a
        # government portal can never be retrieved.  Queuing it anyway burned
        # three job attempts each and then dead-lettered it; drop it at
        # discovery instead.
        if parsed.scheme != "https" or host not in allowed_hosts:
            continue
        clean = canonicalize_discovered_url(absolute)
        if clean in found:
            continue
        hint = "application/pdf" if parsed.path.casefold().endswith(".pdf") else "text/html"
        context = _frame_text(frame)[:300]
        label = anchor_text
        if (not label or _is_generic_label(label)) and context:
            # The row describes the document; the anchor only describes the click.
            label = f"{context} — {label}".strip(" —") if label else context
        found[clean] = DiscoveredLink(
            url=clean,
            label=label[:300],
            content_hint=hint,
            context=context,
            score=score_link(
                url=clean,
                label=label,
                context=context,
                content_hint=hint,
                source=source,
                lexicon=lexicon,
            ),
        )
        if len(found) >= maximum_links:
            break
    return tuple(sorted(found.values(), key=lambda link: (-link.score, link.url)))


def ingest_registered_url(
    *,
    registry: SourceRegistry,
    source_id: str,
    url: str,
    pipeline: IngestionPipeline,
    policy: FetchPolicy | None = None,
    ocr_hook: OcrHook = tesseract_pdf_ocr,
    approved_pdf_sha256s: frozenset[str] = frozenset(),
) -> IngestionOutcome:
    """Fetch and process one registered URL without returning raw source text.

    `approved_pdf_sha256s` no longer gates ordinary PDFs.  It is an override for
    objects above `AUTOMATIC_PDF_PARSE_BYTE_LIMIT`, which are the only ones that
    still require a deliberate operator decision before rendering.
    """

    source = registry.get(source_id)
    if source.id == "idsp_weekly_outbreaks":
        connector = IdspConnector(policy)
        result = (
            connector.fetch_report(url)
            if urllib.parse.urlsplit(url).path.lower().endswith(".pdf")
            else connector.fetch_index(url)
        )
    else:
        result = fetch_url(
            url,
            source_id=source.id,
            allowed_hosts=source.allowed_hosts,
            policy=policy,
        )
    discovered = (
        discover_registered_links(
            result.body,
            index_url=(
                result.receipt.requested_url
                if source.id == "idsp_weekly_outbreaks"
                else result.receipt.final_url
            ),
            source=source,
            lexicon=pipeline.diseases,
        )
        if result.receipt.content_type == "text/html"
        else ()
    )
    if (
        result.receipt.content_type == "application/pdf"
        and result.receipt.byte_length > AUTOMATIC_PDF_PARSE_BYTE_LIMIT
        and result.receipt.sha256 not in approved_pdf_sha256s
    ):
        # Oversized objects are the only ones that still wait for a person.
        return IngestionOutcome(
            receipt=result.receipt,
            signal=None,
            discovered_links=discovered,
            processing_state="metadata_only_pdf_above_automatic_size_limit",
        )
    language_hint = source.languages[0] if len(source.languages) == 1 else None
    parsed = parse_document(
        result.body,
        result.receipt.content_type,
        language_hint=language_hint,
        ocr_hook=ocr_hook,
    )
    if parsed.parser == "pdftotext_layout" and route_unicode(parsed.text) is (
        LanguageRoute.UNDETERMINED
    ):
        # Some official scans carry a text layer built from a broken CID font.
        # It is printable, so the byte-level quality gate accepts it, yet it
        # routes to no language at all. That is the signal to re-read the page
        # with OCR instead of trusting the layer.
        parsed = ocr_hook(result.body, language_hint)
    document_hash = result.receipt.sha256
    document = Document(
        document_id=f"doc_{document_hash[:20]}",
        source_id=source.id,
        canonical_url=result.receipt.final_url,
        retrieved_at=result.receipt.retrieved_at,
        content_type=result.receipt.content_type,
        text=parsed.text,
        sha256=document_hash,
        source_language_hint=language_hint,
        title=parsed.title,
        ocr_confidence=parsed.ocr_confidence,
        article_text=parsed.article_text,
    )
    is_idsp_catalogue = bool(
        source.id == "idsp_weekly_outbreaks"
        and result.receipt.content_type == "application/pdf"
    )
    catalogue_rows = (
        parse_idsp_catalogue_text(parsed.text)
        if is_idsp_catalogue
        else ()
    )
    # A recognized IDSP weekly report is a positive-only event catalogue even
    # when it contains zero Odisha rows.  Falling through to article extraction
    # would turn arbitrary district and disease words in a national report into
    # fabricated Odisha media signals.
    signal = None if is_idsp_catalogue else pipeline.process(
        document, as_of=result.receipt.retrieved_at.date()
    )
    return IngestionOutcome(
        receipt=result.receipt,
        signal=signal,
        discovered_links=discovered,
        catalogue_rows=catalogue_rows,
        processing_state=(
            "positive_only_official_catalogue"
            if is_idsp_catalogue
            else "parsed_redacted_evidence"
        ),
    )
