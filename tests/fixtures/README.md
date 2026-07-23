# Test-fixture policy

The twelve document fixtures are synthetic and visibly watermarked. PDF cases
are OCR/parser surrogates; they exercise deterministic downstream contracts
without copying a government report or news page into a public repository.
`parse_document` is separately tested through an injected OCR hook. The test
suite also generates a raster-only English PDF in memory and runs it through
installed Poppler/Tesseract binaries. That proves process wiring, not Odia or
Hindi accuracy on real government notices; the repository contains no copied
source PDF and no target-corpus OCR benchmark.

The environmental fixture is a sanitised factual excerpt from the NASA POWER
Daily API response retrieved on 21 July 2026 for a fixed demo point. The point
is not asserted to be an authoritative district centroid. Tests compute its
actual SHA-256 at runtime; production creates a new immutable receipt for every
live response.
