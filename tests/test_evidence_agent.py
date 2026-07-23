"""Agent-level correctness: scope, intent and what retrieval actually embeds.

Every case here is a defect that shipped a plausible answer rather than an
error: a statewide figure presented as a district's, an Odisha record used to
answer about a city in another state, a follow-up that silently changed the
question, and a cross-lingual retriever ranking thousands of identical strings.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from packages.contracts.api import LIVE_EVIDENCE_PLACEHOLDER
from packages.nlp import retrieval
from packages.nlp.retrieval import EvidenceRecord
from services.api.evidence_agent import EvidenceAgent
from services.api.main import create_app

STRUCTURED_ONLY = "structured_provenance_only_no_retained_source_text"


def make_client(tmp_path) -> TestClient:
    return TestClient(create_app(f"sqlite:///{tmp_path / 'agent.sqlite3'}"))


def ask(client: TestClient, question: str, **extra) -> dict:
    response = client.post("/api/v1/agent/query", json={"question": question, **extra})
    assert response.status_code == 200
    return dict(response.json()["data"])


def live_row(record_id: str, district_id: str, disease: str, source_id: str, language: str):
    """A live signal row as the database returns it: provenance, no retained text."""

    return {
        "id": record_id,
        "evidence_text": LIVE_EVIDENCE_PLACEHOLDER,
        "district_id": district_id,
        "disease": disease,
        "source_id": source_id,
        "language": language,
        "retrieved_at": "2026-07-21T10:00:00Z",
        "is_fixture": 0,
    }


# ------------------------------------------------------- district-scoped trend


def test_a_district_trend_reports_that_district_not_the_statewide_total(tmp_path) -> None:
    """A district question answered with the state total is wrong by 40x here."""

    from services.api.public_health import malaria_map

    client = make_client(tmp_path)
    rows = malaria_map(metric="total_cases")["records"]
    koraput = next(row for row in rows if row["district_id"] == "OD-DIST-koraput")
    statewide_total = sum(int(row["total_cases"]) for row in rows)

    district = ask(client, "Has malaria in Koraput gone up or down since 2010?")
    assert district["answer_state"] == "official_annual_observation"
    assert district["observation"]["geographic_scope"] == "district"
    assert district["observation"]["district_id"] == "OD-DIST-koraput"
    assert district["observation"]["series"][-1]["total_cases"] == int(
        koraput["total_cases"]
    )
    assert "Koraput" in district["answer_english"]
    assert f"{int(koraput['total_cases']):,}" in district["answer_english"]
    assert f"{statewide_total:,}" not in district["answer_english"]
    assert "DISTRICT_SCOPED_SERIES" in district["reason_codes"]

    state = ask(client, "Has malaria in Odisha gone up or down since 2010?")
    assert state["observation"]["geographic_scope"] == "state"
    assert state["observation"]["district_id"] is None
    assert f"{statewide_total:,}" in state["answer_english"]
    assert "STATEWIDE_SCOPED_SERIES" in state["reason_codes"]


def test_a_trend_question_that_names_years_is_answered_over_those_years(tmp_path) -> None:
    client = make_client(tmp_path)
    windowed = ask(client, "How did malaria change between 2015 and 2024 in Rayagada?")
    years = [row["year"] for row in windowed["observation"]["series"]]
    assert min(years) == 2015
    assert max(years) == 2024
    assert "in 2015 to" in windowed["answer_english"]

    since = ask(client, "Has malaria in Koraput gone up since 2019?")
    assert min(row["year"] for row in since["observation"]["series"]) == 2019


def test_ranking_questions_name_districts_from_the_official_table(tmp_path) -> None:
    from services.api.public_health import malaria_map

    client = make_client(tmp_path)
    rows = malaria_map(metric="total_cases")["records"]
    highest = max(rows, key=lambda row: int(row["total_cases"]))

    for question in (
        "Which district has the most malaria cases?",
        "Show me the malaria heatmap across the state",
        "Rank the districts by malaria burden",
    ):
        payload = ask(client, question)
        assert payload["intent"] == "incidence_request"
        assert payload["answer_state"] == "official_annual_observation"
        assert highest["district_name"] in payload["answer_english"]
        assert f"{int(highest['total_cases']):,}" in payload["answer_english"]
        assert payload["citations"]
        assert "RANKED_BY_REPORTED_TOTAL_NOT_RATE" in payload["reason_codes"]


# ------------------------------------------------------------ places off-scope


def test_a_place_outside_odisha_is_named_not_answered_from_odisha_records(
    tmp_path,
) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")

    outside = ask(client, "How many malaria cases in Bhopal?")
    assert outside["answer_state"] == "district_not_in_odisha"
    assert outside["reason_codes"] == ["PLACE_OUTSIDE_ODISHA_COVERAGE"]
    assert outside["unresolved_place"] == "Bhopal"
    assert outside["evidence"] == []
    assert outside["scope"]["district_id"] is None
    assert "Bhopal" in outside["answer_english"]

    assert ask(client, "Dengue reports in New Delhi?")["answer_state"] == (
        "district_not_in_odisha"
    )


def test_valid_odisha_questions_are_not_refused_as_off_scope_places(tmp_path) -> None:
    """The guard must not fire on interrogatives, on Odisha, or on a real district."""

    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    for question in (
        "Which district has the most malaria cases?",
        "Will dengue outbreak in Odisha in next 3 months?",
        "How many malaria cases were reported in Kandhamal?",
        "Show dengue evidence in Ganjam",
        "Predict malaria risk in Kandhamal for the next 3 months",
        "Is there dengue in Puri District?",
    ):
        assert ask(client, question)["answer_state"] != "district_not_in_odisha", question


# ------------------------------------------------------------- follow-up turns


def test_a_follow_up_inherits_intent_only_when_it_states_none_of_its_own(
    tmp_path,
) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    forecast_history = [
        {"role": "user", "content": "Predict malaria risk in Kandhamal for the next 3 months"},
        {"role": "assistant", "content": "Prior answer text is context, not evidence."},
    ]

    inherited = ask(client, "What about Koraput?", history=forecast_history)
    assert inherited["intent"] == "forecast_request"
    assert inherited["scope"]["district_id"] == "OD-DIST-koraput"
    assert inherited["scope"]["intent_inherited_from_history"] is True
    assert inherited["scope"]["conversation_context_used"] is True

    # The same follow-up with no history is still an evidence search.
    alone = ask(client, "What about Koraput?")
    assert alone["intent"] == "evidence_search"
    assert alone["scope"]["intent_inherited_from_history"] is False

    # A follow-up carrying its own intent keeps it.
    stated = ask(client, "What evidence exists for Koraput?", history=forecast_history)
    assert stated["intent"] == "evidence_search"
    assert stated["scope"]["intent_inherited_from_history"] is False

    counts = ask(
        client,
        "And Koraput?",
        history=[
            {"role": "user", "content": "How many malaria cases were reported in Kandhamal?"},
            {"role": "assistant", "content": "Prior answer text."},
        ],
    )
    assert counts["intent"] == "incidence_request"
    assert counts["answer_state"] == "official_annual_observation"


# ------------------------------------------------------------------- retrieval


def test_live_rows_are_embedded_as_content_not_as_one_repeated_placeholder() -> None:
    """2,108 of 2,119 signals retain no text; the placeholder is not a passage."""

    ganjam = live_row("sig_a", "OD-DIST-ganjam", "dengue", "odisha_hfw_circulars_or", "or")
    koraput = live_row("sig_b", "OD-DIST-koraput", "malaria", "odisha_hfw_news_en", "en")
    text = EvidenceAgent._retrieval_text(ganjam)

    assert LIVE_EVIDENCE_PLACEHOLDER not in text
    assert EvidenceAgent._retained_text(ganjam) == ""
    assert "Dengue" in text
    # The district in all three scripts, so a native-script question matches.
    for name in ("Ganjam", "ଗଞ୍ଜାମ", "गंजाम"):
        assert name in text
    assert "Odia-language" in text
    assert "Odisha Health and Family Welfare circulars (Odia)" in text
    assert text != EvidenceAgent._retrieval_text(koraput)


def test_a_native_script_question_reaches_the_matching_text_free_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordered last on purpose: an untied score, not list order, must pick it."""

    monkeypatch.setenv("ODISHA_NLP_MODE", "off")
    rows = [
        live_row("sig_koraput", "OD-DIST-koraput", "malaria", "odisha_hfw_news_en", "en"),
        live_row("sig_khordha", "OD-DIST-khordha", "cholera", "odisha_hfw_news_en", "en"),
        live_row("sig_ganjam", "OD-DIST-ganjam", "dengue", "odisha_hfw_circulars_or", "or"),
    ]
    records = [
        EvidenceRecord(
            record_id=str(row["id"]),
            text=EvidenceAgent._retrieval_text(row),
            metadata=row,
        )
        for row in rows
    ]
    result = retrieval.rank(
        "ଗଞ୍ଜାମ ଜିଲ୍ଲାରେ ଡେଙ୍ଗୁ",
        records,
        top_k=3,
        document_basis=STRUCTURED_ONLY,
    )
    assert result.ranked[0].record.record_id == "sig_ganjam"
    assert result.ranked[0].score > result.ranked[1].score
    assert result.document_basis == STRUCTURED_ONLY


def test_the_api_declares_what_retrieval_embedded_not_only_which_model(
    tmp_path,
) -> None:
    client = make_client(tmp_path)
    client.post("/api/v1/demo/replay-fixtures")
    payload = ask(client, "Show dengue evidence in Ganjam")
    assert payload["retrieval"]["document_basis"] == (
        "retained_source_text_and_structured_provenance"
    )
