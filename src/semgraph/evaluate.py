import re


def get_entries_from_string(data: str) -> list[str]:
    entries = []
    for e in data.split("\n\n"):
        lines = [line.strip() for line in e.splitlines()]
        lines = [line for line in lines if (line and not line.startswith("#"))]
        string = " ".join(lines)
        string = string.replace("\t", " ")  # replace tabs with a space
        string = re.sub(" +", " ", string)  # squeeze multiple spaces into a single
        if string:
            entries.append(string)
    return entries
