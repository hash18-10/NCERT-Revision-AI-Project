import re


INPUT_FILE = "Understanding_Media_raw.txt"   
OUTPUT_FILE = "understanding_media_clean.txt"

def clean_chapter(text):
    """Remove captions, activity prompts, exercises, and extra spaces."""

    # Remove activity / question prompts
    patterns = [
        r"Look at.*", r"Ask older.*", r"Can you.*", r"Do you think.*",
        r"Pretend.*", r"What.*", r"Why.*", r"How many.*", r"Are the above.*",
        r"Think of.*", r"Find out.*", r"Discuss.*"
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove image captions (short lines < 15 words)
    text = "\n".join([line for line in text.split("\n") if len(line.split()) > 15])

    # Remove EXERCISES and everything after
    text = re.split(r"EXERCISES", text, flags=re.IGNORECASE)[0]

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

if __name__ == "__main__":
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_chapter(raw_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"✅ Cleaned text saved to {OUTPUT_FILE}")
