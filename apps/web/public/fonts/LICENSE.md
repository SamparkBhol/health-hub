# Bundled webfonts

All font files in this directory are redistributed under the SIL Open Font Licence 1.1
(<https://openfontlicense.org>). They are self-hosted so the interface renders identically
offline and never calls a third-party font host at runtime.

| File | Family | Copyright |
| --- | --- | --- |
| `archivo-black-latin.woff2` | Archivo Black | Copyright (c) Omnibus-Type |
| `space-grotesk-latin.woff2` | Space Grotesk | Copyright (c) Florian Karsten |
| `space-mono-400-latin.woff2`, `space-mono-700-latin.woff2` | Space Mono | Copyright (c) Colophon Foundry |
| `noto-sans-oriya.woff2` | Noto Sans Oriya | Copyright (c) The Noto Project Authors |
| `noto-sans-devanagari.woff2` | Noto Sans Devanagari | Copyright (c) The Noto Project Authors |

The Oriya and Devanagari faces are loaded only when a page actually contains those scripts,
via the `unicode-range` descriptor in `src/styles.css`.
