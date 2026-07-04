# TrustyFile Detection Engine v2 — Design

**Date:** 2026-07-04
**Status:** Draft for review
**Sources:** 19-agent audit + research sweep (9 codebase audits, 9 research domains, 1 adversarial completeness critic). Raw reports referenced throughout live in the session audit archive; every codebase claim carries a `file:line` reference there.

---

## 1. Goals

1. **Better detect modified files** — catch text edits, overlays, image pat