# Security policy

## Supported profile

The public competition deployment accepts only synthetic, sanitised or
explicitly approved public-source material. It must not receive IHIP exports,
patient line lists, State credentials or raw personal health data.

Live source text is processed transiently and replaced by a fixed non-content
marker before persistence while multilingual PII-redaction recall remains
unvalidated. Remote PostgreSQL connections are rejected unless the URL sets
`sslmode=verify-full&sslrootcert=system`; loopback PostgreSQL is the local-test
exception.

Outbound production fetches accept only exact registered HTTPS hosts. Each
initial or redirect hop is resolved once, rejected if any answer is non-public,
and connected through an approved numeric address while the original hostname
remains authoritative for the HTTP Host header and TLS certificate/SNI checks.

## Reporting

Do not open a public issue for a suspected vulnerability or exposed personal
data. Contact the repository security owner privately and include the affected
commit, route, reproduction steps and whether any data escaped the redaction
boundary. Replace this paragraph with the sponsor's monitored address before a
public launch.

## Response expectations

- Suspected personal-data exposure: disable the affected source/export first.
- Credential exposure: revoke and rotate immediately; never only delete Git
  history.
- Critical/High security finding: public mutation and live ingestion remain
  disabled until remediation is verified.
- Raw source bodies, credentials and extracted person names must never appear
  in logs, CI artefacts, screenshots or public issue attachments.

The competition profile has no safe-to-host certificate, government approval,
SLA or incident rota. Those are explicit enterprise gates, not implied by this
policy.
