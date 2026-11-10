"""Display helpers for the modification-history timeline (results page).

Pure functions, no web framework involved, so they are unit-testable.
"""

# One displayed change: (old text, new text). Either side can be None:
#   ("1 200,00 EUR", "2 200,00 EUR")  -> a value was replaced
#   ("Remise 10 %", None)             -> a line was removed, nothing took its place
#   (None, "Frais de dossier")        -> a line was added
ChangePair = tuple[str | None, str | None]


def pair_changes(removed: list[str], added: list[str]) -> list[ChangePair]:
    """Pair removed lines with added lines so each reads as "old -> new".

    A PDF revision diff gives two separate lists: the lines that disappeared
    and the lines that appeared. Shown as "- old / + new" they read like a
    developer tool. Pairing them in order (1st removed with 1st added, ...)
    lets the page show "old -> new", which is how a reader thinks about an
    edited amount.

    The pairing is positional, not smart: it does not check that the two
    lines are "about the same thing". That is correct for the common case
    (one value replaced by another) and harmless otherwise, since both texts
    stay visible.

    Args:
        removed: Lines present in the previous version only.
        added: Lines present in the new version only.

    Returns:
        One pair per displayed change. Blank lines are ignored. When one list
        is longer, its extra lines come out with None on the other side.

    Example:
        pair_changes(["1 200,00 EUR", "Remise"], ["2 200,00 EUR"])
        -> [("1 200,00 EUR", "2 200,00 EUR"), ("Remise", None)]
    """
    # Drop blank lines first, so an empty line never "uses up" a partner.
    old_lines = [line for line in removed if line.strip()]
    new_lines = [line for line in added if line.strip()]

    pairs: list[ChangePair] = []
    longest = max(len(old_lines), len(new_lines))
    for i in range(longest):
        old = old_lines[i] if i < len(old_lines) else None
        new = new_lines[i] if i < len(new_lines) else None
        pairs.append((old, new))
    return pairs


def fill_covered_values(pairs: list[ChangePair], covered_by: dict[str, str]) -> list[ChangePair]:
    """Give an "added" line its old value when a cover-up finding knows it.

    When a value is hidden under a box and retyped on top, the old text is
    still in the file, so the revision diff reports an addition only
    ("+ 2 200,00 EUR") and the pair has no old side. The overlay module does
    know what was covered. This joins the two, so the timeline can show
    "1 200,00 EUR -> 2 200,00 EUR".

    Args:
        pairs: Output of pair_changes.
        covered_by: new text -> text it covers, taken from the findings
            (details "replacement" and "covered" of a cover-and-retype flag).

    Returns:
        The same pairs, where (None, new) became (covered_by[new], new)
        whenever `new` is a known replacement. Other pairs are untouched.

    Example:
        fill_covered_values([(None, "2 200,00 EUR")],
                            {"2 200,00 EUR": "1 200,00 EUR"})
        -> [("1 200,00 EUR", "2 200,00 EUR")]
    """
    completed: list[ChangePair] = []
    for old, new in pairs:
        if old is None and new is not None:
            # .strip(): diff lines can carry surrounding spaces, flags do not.
            old = covered_by.get(new.strip())
        completed.append((old, new))
    return completed
