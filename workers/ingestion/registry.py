from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RegistryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SourceSpec:
    id: str
    name: str
    url: str
    allowed_hosts: tuple[str, ...]
    kind: str
    languages: tuple[str, ...]
    enabled: bool
    minimum_interval_seconds: int
    robots_state: str
    rights_state: str
    retention: str
    extra: dict[str, Any]

    @property
    def district_id(self) -> str | None:
        """District this whole route is published by, when the route is district-scoped.

        A collectorate portal at `puri.odisha.gov.in` and a newspaper section at
        `sambad.in/district/puri` are both published *about* one district.  That
        publisher scope is used only as a fallback when the document text itself
        names no district; any district named in the text always wins.
        """

        value = self.extra.get("district_id")
        return str(value) if value else None

    @property
    def content_role(self) -> str:
        """Whether the registered URL is a listing surface or a document itself.

        `index` pages are navigation around other people's documents, so their
        own text is not evidence.  `document` pages (a district health
        department page, a departmental notice) carry the published statement.
        """

        value = str(self.extra.get("content_role", "index")).casefold()
        return value if value in {"index", "document"} else "index"

    @property
    def is_index_only(self) -> bool:
        return self.content_role == "index"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SourceSpec:
        required = {
            "id",
            "name",
            "url",
            "allowed_hosts",
            "kind",
            "languages",
            "enabled",
            "minimum_interval_seconds",
            "robots_state",
            "rights_state",
            "retention",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise RegistryError(f"source is missing required fields: {missing}")
        extras = {key: item for key, item in value.items() if key not in required}
        return cls(
            id=str(value["id"]),
            name=str(value["name"]),
            url=str(value["url"]),
            allowed_hosts=tuple(str(host).lower() for host in value["allowed_hosts"]),
            kind=str(value["kind"]),
            languages=tuple(str(lang) for lang in value["languages"]),
            enabled=bool(value["enabled"]),
            minimum_interval_seconds=int(value["minimum_interval_seconds"]),
            robots_state=str(value["robots_state"]),
            rights_state=str(value["rights_state"]),
            retention=str(value["retention"]),
            extra=extras,
        )


@dataclass(frozen=True, slots=True)
class SourceRegistry:
    schema_version: str
    verified_at: str
    sources: tuple[SourceSpec, ...]

    def get(self, source_id: str, *, require_enabled: bool = True) -> SourceSpec:
        for source in self.sources:
            if source.id == source_id:
                if require_enabled and not source.enabled:
                    raise RegistryError(f"source {source_id!r} is disabled by policy")
                return source
        raise RegistryError(f"unknown source: {source_id!r}")


def load_registry(path: str | Path = "config/sources.yaml") -> SourceRegistry:
    """Load the registry.

    The file is deliberately JSON-compatible YAML, allowing a dependency-free
    parser while remaining valid input for standard YAML tooling.
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = tuple(SourceSpec.from_dict(item) for item in raw.get("sources", []))
    if not sources:
        raise RegistryError("source registry contains no sources")
    ids = [source.id for source in sources]
    if len(ids) != len(set(ids)):
        raise RegistryError("source ids must be unique")
    return SourceRegistry(
        schema_version=str(raw["schema_version"]),
        verified_at=str(raw["verified_at"]),
        sources=sources,
    )
