from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol

import pypdfium2 as pdfium

# Every script this corpus is published in, in Tesseract's own model names.
# Script detection happens inside Tesseract per word; the caller never guesses.
OCR_LANGUAGES: tuple[str, ...] = ("ori", "hin", "eng")


class ParseError(RuntimeError):
    pass


class OcrUnavailable(ParseError):
    pass


def validate_pdf_structure(
    body: bytes,
    *,
    maximum_pages: int = 40,
    maximum_pixels_per_page: int = 25_000_000,
    maximum_total_pixels: int = 200_000_000,
    render_dpi: int = 200,
) -> None:
    """Reject malformed or pathologically large PDFs before external rendering."""

    if not body.startswith(b"%PDF-"):
        raise ParseError("input is not a PDF")
    try:
        document = pdfium.PdfDocument(body)
    except Exception as exc:  # noqa: BLE001 - PDFium errors vary by build
        raise ParseError("PDF structure could not be opened") from exc
    try:
        page_count = len(document)
        if page_count < 1 or page_count > maximum_pages:
            raise ParseError("PDF page count is outside the permitted range")
        total_pixels = 0
        for page_number in range(page_count):
            page = document[page_number]
            try:
                width_points, height_points = page.get_size()
            finally:
                page.close()
            if (
                not math.isfinite(width_points)
                or not math.isfinite(height_points)
                or width_points <= 0
                or height_points <= 0
            ):
                raise ParseError("PDF contains an invalid page size")
            width_pixels = math.ceil(width_points * render_dpi / 72)
            height_pixels = math.ceil(height_points * render_dpi / 72)
            pixels = width_pixels * height_pixels
            if pixels > maximum_pixels_per_page:
                raise ParseError("PDF page exceeds the raster pixel limit")
            total_pixels += pixels
            if total_pixels > maximum_total_pixels:
                raise ParseError("PDF exceeds the total raster pixel limit")
    finally:
        document.close()


@dataclass(frozen=True, slots=True)
class ParsedText:
    text: str
    title: str | None = None
    ocr_confidence: float | None = None
    parser: str = "unknown"
    warnings: tuple[str, ...] = ()
    full_text: str | None = None
    article_text: str | None = None

    @property
    def source_text(self) -> str:
        """Every visible character, including any chrome removed from `text`."""

        return self.full_text if self.full_text is not None else self.text

    @property
    def body_text(self) -> str:
        """The page minus its site chrome, whatever `text` ended up retaining.

        When chrome removal is trusted this is `text`.  When it is not -- the
        origin's own markup left a wrapper unclosed, so removal would have
        emptied the page -- `text` fails open to the whole document while this
        still names the part that is the publication rather than the furniture.
        """

        return self.article_text if self.article_text is not None else self.text


_IGNORED_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})
_BREAKING_TAGS = frozenset({"p", "div", "li", "br", "tr", "h1", "h2", "h3"})
_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
# Repeated site furniture. Removing it stops a bilingual masthead, a language
# switcher or a footer credit from being read as the language, the place or
# the subject of a page.
_CHROME_TAGS = frozenset({"nav", "header", "footer", "aside"})
_CHROME_ROLES = frozenset(
    {"navigation", "banner", "contentinfo", "search", "menu", "menubar", "complementary"}
)
_CHROME_TOKEN = re.compile(
    r"(?:^|[-_ ])(?:nav|navbar|navigation|menu|megamenu|breadcrumb|breadcrumbs|sidebar"
    r"|footer|header|masthead|topbar|skip|skiplink|social|share|cookie|banner"
    r"|language-switcher|langswitch|accessibility)(?:$|[-_ ])"
)


# Chrome is recognised by attribute only on the containers that hold it.  A
# data cell styled `class="menu_style"` is a data cell, and a body carrying
# `layout-sidebar-first` still holds the whole page.
_CHROME_ATTRIBUTE_TAGS = frozenset(
    {"div", "ul", "ol", "nav", "header", "footer", "aside", "section", "form", "table"}
)


def _is_chrome(tag: str, attributes: list[tuple[str, str | None]]) -> bool:
    if tag not in _CHROME_ATTRIBUTE_TAGS:
        return False
    values = {str(key).casefold(): str(value or "") for key, value in attributes}
    if values.get("role", "").casefold() in _CHROME_ROLES:
        return True
    identity = f" {values.get('class', '')} {values.get('id', '')} ".casefold()
    return bool(_CHROME_TOKEN.search(identity))


# A `div class="megamenu-nav"` on the Odisha district portals is left unclosed
# by the origin's own markup, so it swallows the article, the masthead and the
# footer alike.  Site furniture is by definition a minority of a page, so a
# chrome-marked element that ends up owning more than this share of the visible
# letters is treated as an ordinary layout wrapper instead.  Chrome nested
# inside it -- the Chief Minister masthead block, the menus, the breadcrumb --
# is still chrome and is still removed.
MAXIMUM_CHROME_LETTER_SHARE = 0.5


class _VisibleTextParser(HTMLParser):
    """Collect visible text and mark which of it came from site chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        # Index of the innermost chrome element enclosing each chunk, or None.
        self.chrome_owners: list[int | None] = []
        self.title_parts: list[str] = []
        self._open: list[tuple[str, int | None]] = []
        self._chrome_parent: list[int | None] = []
        self._chrome_stack: list[int] = []
        self._ignored_depth = 0
        self._in_title = False

    def _emit(self, data: str, *, chrome: bool | None = None) -> None:
        self.parts.append(data)
        owner = self._chrome_stack[-1] if self._chrome_stack else None
        self.chrome_owners.append(None if chrome is False else owner)

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = tag.casefold()
        if lowered in _IGNORED_TAGS:
            self._ignored_depth += 1
            self._open.append((lowered, None))
            return
        if lowered == "title":
            self._in_title = True
            self._open.append((lowered, None))
            return
        if lowered in _BREAKING_TAGS:
            self._emit("\n", chrome=False)
        if lowered in _VOID_TAGS:
            return
        chrome_id: int | None = None
        if lowered in _CHROME_TAGS or _is_chrome(lowered, attrs):
            chrome_id = len(self._chrome_parent)
            self._chrome_parent.append(
                self._chrome_stack[-1] if self._chrome_stack else None
            )
            self._chrome_stack.append(chrome_id)
        self._open.append((lowered, chrome_id))

    def handle_startendtag(self, tag: str, attrs) -> None:  # noqa: ANN001
        lowered = tag.casefold()
        if lowered in _BREAKING_TAGS:
            self._emit("\n", chrome=False)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered == "title":
            self._in_title = False
        for index in range(len(self._open) - 1, -1, -1):
            name, _ = self._open[index]
            if name != lowered:
                continue
            # Unwind the matched element and anything malformed markup left
            # open inside it, so one stray tag cannot swallow the rest of a page.
            closing = 0
            for unwound_name, unwound_id in self._open[index:]:
                if unwound_id is not None:
                    closing += 1
                if unwound_name in _IGNORED_TAGS and self._ignored_depth:
                    self._ignored_depth -= 1
            if closing:
                # Those ids are, by nesting, the topmost entries of the stack.
                del self._chrome_stack[len(self._chrome_stack) - closing :]
            del self._open[index:]
            return

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        self._emit(data)
        if self._in_title:
            self.title_parts.append(data)

    def chrome_flags(
        self, *, maximum_share: float = MAXIMUM_CHROME_LETTER_SHARE
    ) -> list[bool]:
        """Per-chunk chrome verdicts, ignoring page-sized "chrome" containers.

        A chrome element's weight is all the visible letters it encloses,
        including those attributed to chrome nested inside it, so an unclosed
        wrapper cannot hide how much of the document it claims.
        """

        weight: dict[int, int] = {}
        total = 0
        for chunk, owner in zip(self.parts, self.chrome_owners, strict=False):
            letters = _letters(chunk)
            total += letters
            node = owner
            while node is not None:
                weight[node] = weight.get(node, 0) + letters
                node = self._chrome_parent[node]
        limit = maximum_share * total
        # The innermost owner is the lightest of a chunk's chrome ancestors, so
        # testing it alone answers "is any enclosing chrome element furniture?".
        return [
            owner is not None and weight.get(owner, 0) <= limit
            for owner in self.chrome_owners
        ]


def _clean(text: str) -> str:
    collapsed = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", collapsed).strip()


def _letters(text: str) -> int:
    return sum(character.isalpha() for character in text)


def parse_html(
    body: bytes,
    *,
    encoding: str = "utf-8",
    strip_chrome: bool = True,
    minimum_retained_characters: int = 120,
    minimum_retained_share: float = 0.15,
) -> ParsedText:
    """Extract visible text, dropping repeated navigation/header/footer chrome.

    Chrome removal is fail-open: if it would leave too little of the page, the
    full text is kept instead, because a site whose body is inside a `header`
    element must not silently become an empty document.

    The masthead of every `odisha.gov.in` portal names the Chief Minister.  It
    is chrome, so it is removed here rather than travelling into the document
    body, where the PII redactor would correctly flag the name and put an
    otherwise ordinary health notice into privacy review.
    """

    source = body.decode(encoding, errors="replace")
    parser = _VisibleTextParser()
    parser.feed(source)
    chrome_flags = parser.chrome_flags()
    full_text = _clean("".join(parser.parts))
    title = " ".join("".join(parser.title_parts).split()) or None
    warnings: tuple[str, ...] = ()
    text = full_text
    article_text: str | None = None
    if strip_chrome and any(chrome_flags):
        main_text = _clean(
            "".join(
                chunk
                for chunk, chrome in zip(parser.parts, chrome_flags, strict=False)
                if not chrome
            )
        )
        article_text = main_text
        retained = _letters(main_text)
        if retained >= minimum_retained_characters and retained >= (
            minimum_retained_share * max(1, _letters(full_text))
        ):
            text = main_text
            warnings = ("site_chrome_removed_before_extraction",)
        else:
            warnings = ("site_chrome_retained_after_low_yield_removal",)
    return ParsedText(
        text=text,
        title=title,
        parser="stdlib_html",
        warnings=warnings,
        full_text=full_text,
        article_text=article_text,
    )


class OcrHook(Protocol):
    def __call__(self, body: bytes, language_hint: str | None = None) -> ParsedText: ...


def pdftotext_extract(body: bytes, *, timeout_seconds: int = 60) -> ParsedText:
    """Extract a trustworthy text layer when one exists, retaining layout."""

    if not body.startswith(b"%PDF-"):
        raise ParseError("input is not a PDF")
    executable = shutil.which("pdftotext")
    if executable is None:
        raise OcrUnavailable("pdftotext is not installed")
    with tempfile.TemporaryDirectory(prefix="health-pdf-text-") as directory:
        pdf_path = Path(directory) / "input.pdf"
        pdf_path.write_bytes(body)
        try:
            completed = subprocess.run(  # noqa: S603
                [executable, "-layout", str(pdf_path), "-"],
                check=True,
                capture_output=True,
                timeout=timeout_seconds,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ParseError("pdftotext failed") from exc
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    printable = sum(character.isprintable() or character.isspace() for character in text)
    quality = printable / max(1, len(text))
    if len(text) < 50 or quality < 0.95 or text.count("�") > max(2, len(text) // 100):
        raise ParseError("PDF text layer is absent or low quality")
    return ParsedText(text=text, parser="pdftotext_layout")


def tesseract_languages(
    *, which: Callable[[str], str | None] = shutil.which
) -> str:
    """Return the Tesseract `-l` value: every installed model we can read.

    The route's language field says what a *site* mostly publishes, not what a
    given scan contains: an Odia district portal routinely posts English
    circulars and Hindi central advisories.  Recognising an English scan with
    the Odia model alone produced fluent-looking Odia that was never on the
    page.  Passing all three models lets Tesseract choose per word, and any
    model absent from this image is dropped rather than failing the run.
    """

    installed: list[str] = []
    if which("tesseract") is not None:
        try:
            listing = subprocess.run(  # noqa: S603
                [str(which("tesseract")), "--list-langs"],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            listing = ""
        installed = [line.strip() for line in listing.splitlines()[1:] if line.strip()]
    available = [code for code in OCR_LANGUAGES if code in installed]
    return "+".join(available or OCR_LANGUAGES)


def tesseract_pdf_ocr(
    body: bytes,
    language_hint: str | None = None,  # noqa: ARG001 - route hint is not trusted
    *,
    maximum_pages: int = 8,
    timeout_seconds: int = 240,
) -> ParsedText:
    """OCR a small PDF inside one wall-clock budget below the 300 s job lease.

    `language_hint` is accepted for interface compatibility and deliberately
    ignored; see `tesseract_languages`.
    """

    if not body.startswith(b"%PDF-"):
        raise ParseError("input is not a PDF")
    if maximum_pages < 1 or maximum_pages > 8:
        raise ParseError("OCR page limit must be between 1 and 8")
    # Even an accidental larger caller value cannot consume the whole lease.
    budget_seconds = min(max(1, timeout_seconds), 240)
    started_at = time.monotonic()

    def remaining_seconds() -> float:
        remaining = budget_seconds - (time.monotonic() - started_at)
        if remaining <= 0:
            raise ParseError("OCR wall-clock budget exceeded")
        return remaining

    validate_pdf_structure(
        body,
        maximum_pages=maximum_pages,
        maximum_total_pixels=150_000_000,
    )
    pdftoppm = shutil.which("pdftoppm")
    tesseract = shutil.which("tesseract")
    if pdftoppm is None:
        raise OcrUnavailable("pdftoppm is not installed")
    if tesseract is None:
        raise OcrUnavailable("tesseract is not installed")
    language = tesseract_languages()
    with tempfile.TemporaryDirectory(prefix="health-ocr-") as directory:
        root = Path(directory)
        pdf_path = root / "input.pdf"
        pdf_path.write_bytes(body)
        prefix = root / "page"
        # Executable paths come from shutil.which, all arguments are a fixed
        # list, the input is an isolated temporary file, and no shell is used.
        try:
            subprocess.run(  # noqa: S603
                [
                    pdftoppm,
                    "-f",
                    "1",
                    "-l",
                    str(maximum_pages),
                    "-r",
                    "200",
                    "-png",
                    str(pdf_path),
                    str(prefix),
                ],
                check=True,
                capture_output=True,
                timeout=remaining_seconds(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ParseError("OCR wall-clock budget exceeded during rendering") from exc
        except subprocess.CalledProcessError as exc:
            raise ParseError("PDF renderer failed") from exc
        pages = sorted(root.glob("page-*.png"))
        if not pages:
            raise ParseError("PDF renderer emitted no pages")
        texts: list[str] = []
        confidences: list[float] = []
        for page in pages:
            try:
                output = subprocess.run(  # noqa: S603
                    [tesseract, str(page), "stdout", "-l", language, "--psm", "6", "tsv"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=remaining_seconds(),
                ).stdout
            except subprocess.TimeoutExpired as exc:
                raise ParseError("OCR wall-clock budget exceeded during recognition") from exc
            except subprocess.CalledProcessError as exc:
                raise ParseError("Tesseract recognition failed") from exc
            words: list[str] = []
            for line in output.splitlines()[1:]:
                columns = line.split("\t")
                if len(columns) < 12 or not columns[11].strip():
                    continue
                words.append(columns[11].strip())
                try:
                    confidence = float(columns[10])
                except ValueError:
                    continue
                if confidence >= 0:
                    confidences.append(confidence / 100.0)
            texts.append(" ".join(words))
    text = "\n".join(texts).strip()
    if not text:
        raise ParseError("OCR produced no text")
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return ParsedText(
        text=text,
        ocr_confidence=confidence,
        parser="tesseract_pdf_ocr",
        warnings=("ocr_quality_unvalidated_on_odia_government_documents",),
    )


def parse_document(
    body: bytes,
    content_type: str,
    *,
    language_hint: str | None = None,
    force_ocr: bool = False,
    ocr_hook: OcrHook = tesseract_pdf_ocr,
    validate_structure: bool = True,
) -> ParsedText:
    if content_type == "text/html":
        return parse_html(body)
    if content_type == "text/plain":
        return ParsedText(text=body.decode("utf-8", errors="replace"), parser="plain_text")
    if content_type == "application/pdf":
        if validate_structure:
            validate_pdf_structure(body)
        if force_ocr:
            return ocr_hook(body, language_hint)
        try:
            return pdftotext_extract(body)
        except (OcrUnavailable, ParseError):
            # Some official scans carry a short garbage text layer. Low-quality
            # extraction falls through to OCR rather than being trusted.
            return ocr_hook(body, language_hint)
    raise ParseError(f"unsupported content type: {content_type}")
