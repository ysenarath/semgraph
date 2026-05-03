import re


def get_entries_from_string(data: str) -> list[str]:
    entries = []
    for e in data.split("\n\n"):
        lines = [l.strip() for l in e.splitlines()]
        lines = [l for l in lines if (l and not l.startswith("#"))]
        string = " ".join(lines)
        string = string.replace("\t", " ")  # replace tabs with a space
        string = re.sub(" +", " ", string)  # squeeze multiple spaces into a single
        if string:
            entries.append(string)
    return entries
