"""Objective 1 regression tests: trilingual routing, extraction and discovery.

Every fixture string in this module is a short structural excerpt of a page
that was fetched live from the registered source on 2026-07-21 (mastheads,
menu labels and notice titles), so the behaviour under test is the behaviour
seen on the real sites rather than an invented shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from workers.ingestion.connectors import discover_registered_links, ingest_registered_url
from workers.ingestion.diseases import DiseaseLexicon
from workers.ingestion.geography import DistrictGazetteer
from workers.ingestion.language import LanguageRoute, route_unicode, script_profile
from workers.ingestion.models import FetchReceipt, FetchResult
from workers.ingestion.parse import ParsedText, parse_html
from workers.ingestion.pipeline import IngestionPipeline
from workers.ingestion.registry import load_registry

# health.odisha.gov.in/en/notifications/circulars — the Odia string in the
# masthead is site chrome on every page of the state CMS, including English.
ENGLISH_PAGE_WITH_ODIA_MASTHEAD = """Circulars | Department of Health & Family Welfare
 Skip to main content
 Government of Odisha
 ଓଡ଼ିଶା ସରକାର
 Login
 Register
 Search
 English
 Odia
Circulars
 Sr. No.
 Title
 Date
 Details/Download
 8
 Leprosy to be identified as " Reportable Disease " in " State of Odisha "
 22/12/2023
 Download(1.74 MB)
 10
 Increase in surveillance of ILI/SARI
 28/11/2023
 Download(138.94 KB)
"""

# health.odisha.gov.in/or/notifications/circulars — same template, Odia body.
ODIA_PAGE_WITH_ENGLISH_CHROME = """ପରିପତ୍ର ଏବଂ ବିଜ୍ଞପ୍ତି | Department of Health & Family Welfare
 Skip to main content
 Government of Odisha
 ଓଡ଼ିଶା ସରକାର
 Login
 Register
ପରିପତ୍ର ଏବଂ ବିଜ୍ଞପ୍ତି
 କ୍ରମିକ ସଂଖ୍ୟା
 ବିଷୟ
 ତାରିଖ
 1
 "କର୍କଟ" ହେଉଛି 'ଓଡିଶା ରାଜ୍ୟରେ' ଏକ "ରିପୋର୍ଟଯୋଗ୍ୟ ରୋଗ"
 20/10/2022
 Download(674.76 KB)
 2
 ୨୦୨୨ ରେ ପୋଷ୍ଟ ମୌସୁମୀ ଘୂର୍ଣ୍ଣିବଳୟ ପରିଚାଳନା ପାଇଁ ପରାମର୍ଶଦାତା
 18/10/2022
 5
 ଚୟନ ଗ୍ରେଡ୍ ରାଙ୍କରେ ମେଡିକାଲ୍ ଅଧିକାରୀଙ୍କ ପଦୋନ୍ନତି ଏବଂ ପୋଷ୍ଟିଂ
 23/12/2016
 6
 ମେଡିକାଲ ଅଫିସରଙ୍କ ସ୍ଥାନାନ୍ତର ଏବଂ ପୋଷ୍ଟିଂ
 23/12/2016
 8
 ଓଡିଶା ମେଡିକାଲ୍ ଶିକ୍ଷା ସେବା ନିୟମ -୨୦୧୩
 18/12/2013
 9
 ରାଜ୍ୟର ସରକାରୀ ଆୟୁର୍ବେଦ କଲେଜଗୁଡ଼ିକର ବିଜ୍ଞପ୍ତି
 06/09/2013
 10
 ଓଡିଶା ଫାର୍ମାସିଷ୍ଟ ସେବା କ୍ୟାଡରର ପୁନଃନିର୍ମାଣ
 24/08/2013
"""

# ncvbdc.mohfw.gov.in/index.php?lang=2 — the only live Hindi route that fetches.
HINDI_PAGE = """होम :: राष्ट्रीय वेक्टर जनित रोग नियंत्रण कार्यक्रम
 नेविगेशन छोड़ें
 स्क्रीन रीडर प्रयोग
 English
 भारत सरकार
 स्‍वास्‍थ्‍य और परिवार कल्‍याण विभाग
 समाचार और प्रमुख विशेषताएं
 चिकनगुनिया के लिए राष्ट्रीय दिशानिर्देश ( प्रकशित करने की तारीख :23/09/2016 ) [PDF] [6276 KB]
 वेब सूचना प्रबंधक
"""

# ganjam.odisha.gov.in/or — an Odia district portal whose notice titles are
# published in English. This page is genuinely bilingual, not chrome-skewed.
BILINGUAL_DISTRICT_PAGE = """ମୂଳପୃଷ୍ଠା | Ganjam
 ଗଞ୍ଜାମ ଜିଲ୍ଲା ଭୋଟର ସଚେତନତା ପ୍ରତିଯୋଗିତା
 ଜିଲ୍ଲା ଏକ ନଜର ରେ
 ନୂତନ ତଥ୍ୟ
 ବିଜ୍ଞପ୍ତି
 ଟେଣ୍ଡର
 ନିଯୁକ୍ତି
 ଡାକ୍ତରଖାନା / ଚିକିତ୍ସାଳୟ
 ଗଞ୍ଜାମ ଜିଲ୍ଲା ତା ୦୧.୦୪.୧୯୩୬ ରିଖରେ ତାର ସ୍ଥିତିକୁ ଆସିଥିଲା । ଋଷିକୂଲ୍ୟା ନଦୀର ଉତ୍ତର ପାର୍ଶ୍ବରେ ଥିବା ୟୁରୋପୀୟ ଦୁର୍ଗ
 ଗଞ୍ଜାମ ନାମକ କ୍ଷୁଦ୍ର ସହରର ନାମାନୁସାରେ ଏହି ଜିଲ୍ଲା ନାମିତ ଯାହା ପୂର୍ବେ ଏହି ଜିଲ୍ଲାର ରାଜଧାନୀ ଥିଲା ।
 General Notice for col-2 correction cases for 2025 under Digapahandi Tahasil
 TENDER CALL NOTICE - KASTURBA GANDHI BALIKA VIDYALAYA, BEGUNIAPADA, GANJAM
 Notice for the time of Examination for the different Posts under NHM Ganjam
 ONLINE REGISTRATION FOR MALE CANDIDATES FOR SELECTION TEST TO JOIN THE IAF
"""


@pytest.fixture(scope="module")
def lexicon() -> DiseaseLexicon:
    return DiseaseLexicon.load()


def test_english_page_carrying_an_odia_masthead_routes_english() -> None:
    assert route_unicode(ENGLISH_PAGE_WITH_ODIA_MASTHEAD) is LanguageRoute.ENGLISH
    profile = script_profile(ENGLISH_PAGE_WITH_ODIA_MASTHEAD)
    # The Odia chrome is real, it is simply a rounding error of the page.
    assert profile.counts["or"] > 0
    assert profile.share("or") < 0.05


def test_odia_page_carrying_english_chrome_routes_odia() -> None:
    assert route_unicode(ODIA_PAGE_WITH_ENGLISH_CHROME) is LanguageRoute.ODIA


def test_hindi_official_page_routes_hindi() -> None:
    assert route_unicode(HINDI_PAGE) is LanguageRoute.HINDI


def test_genuinely_bilingual_district_page_is_mixed_not_silently_forced() -> None:
    assert route_unicode(BILINGUAL_DISTRICT_PAGE) is LanguageRoute.MIXED


def test_a_masthead_sized_script_minority_cannot_flip_the_route() -> None:
    # The previous absolute two-character rule routed any page containing
    # "ଓଡ଼ିଶା ସରକାର" to Odia. Share-based routing must not.
    english = (
        "Government of Odisha\nଓଡ଼ିଶା ସରକାର\n"
        + "Dengue cases were reported in the district health bulletin.\n" * 4
    )
    hindi_with_odia_chrome = (
        "ଓଡ଼ିଶା ସରକାର\n"
        + "गंजाम जिले में डेंगू के मामले दर्ज किए गए और स्वास्थ्य विभाग ने निगरानी बढ़ाई।\n" * 3
    )
    assert route_unicode(english) is LanguageRoute.ENGLISH
    assert route_unicode(hindi_with_odia_chrome) is LanguageRoute.HINDI


def test_romanised_indic_and_tiny_inputs_stay_undetermined() -> None:
    assert route_unicode("Khordha jillare dengu mamla chihnata heichi.") is (
        LanguageRoute.UNDETERMINED
    )
    assert route_unicode("A") is LanguageRoute.UNDETERMINED


def test_chrome_is_stripped_before_extraction_but_never_empties_a_page() -> None:
    body = (
        b"<html><body>"
        b"<header><p>Government of Odisha</p><p>&#2835;&#2849;&#2879;&#2936;&#2878;</p></header>"
        b"<nav class='main-menu'><ul><li>Home</li><li>Contact Us</li></ul></nav>"
        b"<main><table><tr><td>1</td><td>Increase in surveillance of ILI/SARI</td></tr>"
        b"<tr><td>2</td><td>Dengue advisory for Khordha district hospitals</td></tr>"
        b"<tr><td>3</td><td>Leprosy to be identified as a reportable disease in Odisha</td></tr>"
        b"<tr><td>4</td><td>Acute diarrhoeal disease preparedness before the monsoon</td></tr>"
        b"</table></main>"
        b"<footer><p>Powered by OCAC</p></footer>"
        b"</body></html>"
    )
    parsed = parse_html(body)
    assert "Increase in surveillance of ILI/SARI" in parsed.text
    assert "Powered by OCAC" not in parsed.text
    assert "Powered by OCAC" in parsed.source_text
    assert parsed.warnings == ("site_chrome_removed_before_extraction",)

    only_chrome = b"<html><body><header><p>Dengue advisory for Khordha</p></header></body></html>"
    fallback = parse_html(only_chrome)
    assert "Dengue advisory for Khordha" in fallback.text
    assert fallback.warnings == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dengue fever cases confirmed in Khordha", "dengue"),
        ("ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁର ମାମଲା ବୃଦ୍ଧି", "dengue"),
        ("गंजाम में डेंगू के मामले बढ़े", "dengue"),
        ("Malaria elimination review", "malaria"),
        ("ମ୍ୟାଲେରିଆ ନିୟନ୍ତ୍ରଣ କାର୍ଯ୍ୟକ୍ରମ", "malaria"),
        ("मलेरिया उन्मूलन कार्यक्रम", "malaria"),
        ("Acute diarrhoeal disease reported", "acute_diarrhoeal_disease"),
        ("ରାୟଗଡ଼ାରେ ଅତିସାର ମାମଲା", "acute_diarrhoeal_disease"),
        ("तीव्र अतिसार के मामले", "acute_diarrhoeal_disease"),
        ("Cholera outbreak investigation", "cholera"),
        ("କୋରାପୁଟରେ ହଇଜା ପ୍ରକୋପ", "cholera"),
        ("हैजा का प्रकोप", "cholera"),
        ("Chikungunya surveillance", "chikungunya"),
        ("ଚିକୁନଗୁନିଆ ମାମଲା", "chikungunya"),
        ("चिकनगुनिया के लिए राष्ट्रीय दिशानिर्देश", "chikungunya"),
        ("Acute encephalitis syndrome ward", "aes_je"),
        ("ଜାପାନୀ ଏନସେଫାଲାଇଟିସ ଟୀକାକରଣ", "aes_je"),
        ("जापानी इंसेफेलाइटिस टीकाकरण", "aes_je"),
        ("Scrub typhus positive samples", "scrub_typhus"),
        ("ସ୍କ୍ରବ ଟାଇଫସ ପରୀକ୍ଷା", "scrub_typhus"),
        ("स्क्रब टाइफस की जांच", "scrub_typhus"),
        ("Leptospirosis after floods", "leptospirosis"),
        ("ଲେପ୍ଟୋସ୍ପାଇରୋସିସ ସତର୍କତା", "leptospirosis"),
        ("लेप्टोस्पायरोसिस की चेतावनी", "leptospirosis"),
        ("Acute hepatitis A cluster", "hepatitis"),
        ("ଜଣ୍ଡିସ ପ୍ରକୋପ", "hepatitis"),
        ("पीलिया के मरीज", "hepatitis"),
        ("Typhoid cases in the block", "typhoid"),
        ("ଟାଇଫଏଡ ଜ୍ୱର", "typhoid"),
        ("टाइफाइड के मामले", "typhoid"),
        ("Measles-rubella campaign", "measles"),
        ("ମିଳିମିଳା ଟୀକା", "measles"),
        ("खसरा टीकाकरण अभियान", "measles"),
        ("Increase in surveillance of ILI/SARI", "influenza_h1n1"),
        ("H1N1 positive cases", "influenza_h1n1"),
        ("ସ୍ୱାଇନ ଫ୍ଲୁ ସତର୍କତା", "influenza_h1n1"),
        ("स्वाइन फ्लू की चेतावनी", "influenza_h1n1"),
        ("Snakebite deaths compensation", "snakebite"),
        ("ସର୍ପଦଂଶରେ ମୃତ୍ୟୁ", "snakebite"),
        ("सर्पदंश से मौत", "snakebite"),
        ("Heat stroke admissions", "heat_illness"),
        ("ହିଟ ଷ୍ଟ୍ରୋକ ରୋଗୀ", "heat_illness"),
        ("लू लगने से बीमार", "heat_illness"),
    ],
)
def test_required_disease_groups_match_in_all_three_languages(
    lexicon: DiseaseLexicon, text: str, expected: str
) -> None:
    assert expected in lexicon.find(text), text


def test_lexicon_covers_every_required_phase_one_group_in_three_scripts(
    lexicon: DiseaseLexicon,
) -> None:
    required = {
        "dengue",
        "malaria",
        "acute_diarrhoeal_disease",
        "cholera",
        "chikungunya",
        "aes_je",
        "scrub_typhus",
        "leptospirosis",
        "hepatitis",
        "typhoid",
        "measles",
        "influenza_h1n1",
        "snakebite",
        "heat_illness",
    }
    assert required <= set(lexicon.terms)
    for disease in required:
        terms = lexicon.terms[disease]
        assert any(any(0x0B00 <= ord(c) <= 0x0B7F for c in term) for term in terms), disease
        assert any(any(0x0900 <= ord(c) <= 0x097F for c in term) for term in terms), disease
        assert any(all(ord(c) < 0x0300 for c in term) for term in terms), disease


@pytest.mark.parametrize(
    "text",
    [
        "The hospital will add beds in Ganjam next week.",
        "यह दस्तावेज़ स्वास्थ्य विभाग का है।",
        "She wore a red sari to the district function.",
        "Download(2.34 MB) tender document",
    ],
)
def test_lexicon_does_not_fire_on_ordinary_words(lexicon: DiseaseLexicon, text: str) -> None:
    assert lexicon.find(text) == ()


def test_disease_matches_report_the_surface_term_for_review(lexicon: DiseaseLexicon) -> None:
    assert lexicon.find_terms("ଖୋର୍ଦ୍ଧାରେ ଡେଙ୍ଗୁର ମାମଲା") == (("dengue", "ଡେଙ୍ଗୁ"),)


def test_gazetteer_resolves_native_and_variant_spellings() -> None:
    gazetteer = DistrictGazetteer.load()
    resolved = {
        match.district_id
        for match in gazetteer.resolve(
            "Cases in Kendrapada, Baleswar, Khorda, କେଉଁଝର and ଫୁଲବାଣୀ"
        )
    }
    assert resolved == {
        "OD-DIST-kendrapara",
        "OD-DIST-balasore",
        "OD-DIST-khordha",
        "OD-DIST-keonjhar",
        "OD-DIST-kandhamal",
    }


@pytest.mark.parametrize(
    ("layer_text", "expect_ocr"),
    [
        # A broken CID font maps glyphs to the wrong code points: printable,
        # so the byte-level quality gate accepts it, but no language at all.
        (
            "Jryhuqphqw ri Rglvkd Khdowk Ghsduwphqw Qrwlilfdwlrq Qr 23810 "
            "gdwhg 20 Rfwrehu 2022 uhjduglqj d uhsruwdeoh glvhdvh",
            True,
        ),
        # A real English text layer must be trusted instead of re-OCRed.
        (
            "GOVERNMENT OF ODISHA HEALTH & FAMILY WELFARE DEPARTMENT NOTIFICATION "
            "cancer is hereby declared a reportable disease in the State of Odisha.",
            False,
        ),
    ],
)
def test_unroutable_pdf_text_layer_falls_back_to_ocr(
    monkeypatch: pytest.MonkeyPatch, layer_text: str, expect_ocr: bool
) -> None:
    body = b"%PDF-1.4 pretend"
    receipt = FetchReceipt(
        source_id="odisha_hfw_circulars_or",
        requested_url="https://health.odisha.gov.in/x.pdf",
        final_url="https://health.odisha.gov.in/x.pdf",
        retrieved_at=datetime(2026, 7, 21, tzinfo=UTC),
        status_code=200,
        content_type="application/pdf",
        byte_length=len(body),
        sha256="a" * 64,
    )
    monkeypatch.setattr(
        "workers.ingestion.connectors.fetch_url",
        lambda *args, **kwargs: FetchResult(receipt=receipt, body=body),  # noqa: ARG005
    )
    monkeypatch.setattr(
        "workers.ingestion.connectors.parse_document",
        lambda *args, **kwargs: ParsedText(text=layer_text, parser="pdftotext_layout"),  # noqa: ARG005
    )
    calls: list[str | None] = []

    def ocr_hook(payload: bytes, language_hint: str | None = None) -> ParsedText:
        calls.append(language_hint)
        return ParsedText(
            text="ଓଡ଼ିଶା ରାଜ୍ୟରେ କର୍କଟ ଏକ ରିପୋର୍ଟଯୋଗ୍ୟ ରୋଗ ଭାବେ ଘୋଷିତ ହୋଇଛି।",
            ocr_confidence=0.8,
            parser="stub_ocr",
        )

    outcome = ingest_registered_url(
        registry=load_registry(),
        source_id="odisha_hfw_circulars_or",
        url="https://health.odisha.gov.in/x.pdf",
        pipeline=IngestionPipeline.default(),
        ocr_hook=ocr_hook,
        approved_pdf_sha256s=frozenset({"a" * 64}),
    )
    assert bool(calls) is expect_ocr
    assert outcome.signal is not None
    assert outcome.signal.diseases == ("cancer",)
    assert calls == (["or"] if expect_ocr else [])


def test_index_connector_ranks_notice_rows_above_site_furniture() -> None:
    # The row layout mirrors health.odisha.gov.in: the anchor says only
    # "Download(1.74 MB)" while the row carries the subject of the circular.
    body = b"""<html><body>
    <nav><a href="/screen-reader">Screen Reader Access</a>
    <a href="/en/sitemap">Sitemap</a></nav>
    <table>
      <tr>
        <td>8</td>
        <td>Leprosy to be identified as " Reportable Disease " in " State of Odisha "</td>
        <td>22/12/2023</td>
        <td><a href="/sites/default/files/2023-12/31780.PDF">Download(1.74 MB)</a></td>
      </tr>
      <tr>
        <td>17</td>
        <td>Training of Govt. Doctors for One Year PGDPHM Course</td>
        <td>24/05/2023</td>
        <td><a href="/sites/default/files/2023-05/pgdphm.pdf">Download(419.44 KB)</a></td>
      </tr>
    </table>
    <a href="/user/login">Login</a>
    </body></html>"""
    source = load_registry().get("odisha_hfw_circulars_en")
    links = discover_registered_links(
        body,
        index_url="https://health.odisha.gov.in/en/notifications/circulars",
        source=source,
        lexicon=DiseaseLexicon.load(),
    )
    urls = [link.url for link in links]
    assert urls[0] == "https://health.odisha.gov.in/sites/default/files/2023-12/31780.PDF"
    assert "Leprosy" in links[0].label
    assert links[0].content_hint == "application/pdf"
    assert links[0].score > links[1].score
    # Accessibility and session surfaces must never outrank a notice row.
    assert urls.index("https://health.odisha.gov.in/user/login") > urls.index(
        "https://health.odisha.gov.in/sites/default/files/2023-05/pgdphm.pdf"
    )
    assert links[-1].score < 0
