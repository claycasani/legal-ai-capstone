import re

import tiktoken

ENCODING = None

CLAUSE_HEADER_PATTERN = re.compile(
    r"(?im)^(section\s+\d+(\.\d+)*|article\s+[ivx\d]+|"
    r"\d+(\.\d+)*\s+[A-Z][^\n]*|[A-Z][A-Z \-/]{4,})$"
)

CLAUSE_PATTERNS = {
    "Payment / financial terms": [
        "payment", "fees", "rent", "deposit", "compensation", "salary", "invoice", "charges",
    ],
    "Termination / cancellation": [
        "terminate", "termination", "cancel", "expiration", "renewal", "breach",
    ],
    "Liability / indemnification": [
        "liability", "indemnify", "indemnification", "damages", "losses",
    ],
    "Confidentiality / non-disclosure": [
        "confidential", "confidentiality", "non-disclosure", "nda", "proprietary information",
    ],
    "IP ownership / licensing": [
        "intellectual property", "ip ownership", "license", "licensing", "copyright", "patent",
    ],
    "Governing law / jurisdiction": [
        "governing law", "jurisdiction", "venue", "state of", "laws of",
    ],
    "Dispute resolution / arbitration": [
        "dispute", "arbitration", "arbitrate", "litigation", "court", "jury trial",
    ],
    "Warranties / representations": [
        "warranty", "warranties", "representations", "as is", "disclaimer",
    ],
}


def count_tokens(text: str) -> int:
    global ENCODING
    try:
        if ENCODING is None:
            ENCODING = tiktoken.get_encoding("cl100k_base")
        return len(ENCODING.encode(text))
    except Exception:
        # Keeps local/dev deployments usable when the tiktoken cache is cold and
        # network access is unavailable. Contracts average roughly 1.3 tokens per word.
        return max(1, int(len(text.split()) * 1.3))


def split_into_sections(text: str) -> list[str]:
    sections = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if CLAUSE_HEADER_PATTERN.match(stripped) and current:
            sections.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def build_chunk(words: list[str], chunk_index: int) -> dict:
    chunk_text = " ".join(words).strip()
    return {
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "token_count": count_tokens(chunk_text),
        "clause_tags": tag_chunk_text(chunk_text),
    }


def get_overlap_words(words: list[str], target_overlap_tokens: int = 175) -> list[str]:
    overlap_words = []
    for word in reversed(words):
        overlap_words.insert(0, word)
        if count_tokens(" ".join(overlap_words)) >= target_overlap_tokens:
            break
    return overlap_words


def chunk_document(text: str, chunk_size: int = 900, overlap: int = 175) -> list[dict]:
    chunks = []
    current_words = []
    chunk_index = 0

    for section in split_into_sections(text):
        section_words = section.split()
        trial_words = current_words + section_words

        if count_tokens(" ".join(trial_words)) <= chunk_size:
            current_words = trial_words
            continue

        if current_words:
            chunks.append(build_chunk(current_words, chunk_index))
            chunk_index += 1
            current_words = get_overlap_words(current_words, overlap) + section_words
            continue

        temp_words = section_words[:]
        while temp_words:
            candidate = []
            while temp_words and count_tokens(" ".join(candidate + [temp_words[0]])) <= chunk_size:
                candidate.append(temp_words.pop(0))
            if not candidate:
                candidate = [temp_words.pop(0)]
            chunks.append(build_chunk(candidate, chunk_index))
            chunk_index += 1
            overlap_words = get_overlap_words(candidate, overlap)
            temp_words = overlap_words + temp_words if temp_words else temp_words
            if len(temp_words) == len(overlap_words):
                break

    if current_words:
        chunks.append(build_chunk(current_words, chunk_index))

    return chunks


def tag_chunk_text(text: str) -> list[str]:
    lowered = text.lower()
    tags = []
    for tag, patterns in CLAUSE_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            tags.append(tag)
    return tags
