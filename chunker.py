"""
Paragraph-aware, TOKEN-based semantic text chunker.

Uses the embedding model's actual tokenizer to count tokens accurately,
avoiding the silent truncation bug where word-count approximations
(~1.3 tokens/word) exceed BGE-small's 512-token hard limit.

Strategy:
  1. Split on paragraph boundaries (\\n\\n)
  2. Split oversized paragraphs at sentence boundaries
  3. Accumulate into chunks of ~max_tokens tokens
  4. Merge tiny trailing chunks
  5. Inject overlap between neighboring chunks
"""
import re
from typing import List, Dict, Optional

import config


class FastTokenizerAdapter:
    """
    Adapter around the local ONNX Tokenizer instance providing
    .encode() and .decode() methods matching AutoTokenizer's interface.
    """
    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids)


# Lazily loaded tokenizer singleton
_tokenizer = None


def get_tokenizer():
    """
    Load the tokenizer directly from FastEmbed's local ONNX model instance.
    Zero network requests to Hugging Face Hub at runtime!
    """
    global _tokenizer
    if _tokenizer is None:
        from embeddings import get_embedding_model
        model = get_embedding_model()
        _tokenizer = FastTokenizerAdapter(model.tokenizer)
    return _tokenizer


def _count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the model's actual tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_into_sentences(text: str) -> List[str]:
    """Split text on sentence terminators (. ? !) while keeping punctuation."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if s.strip()]


def _split_oversized_paragraph(
    paragraph: str, max_tokens: int, tokenizer
) -> List[str]:
    """Break an oversized paragraph into sentence-sized blocks."""
    sentences = _split_into_sentences(paragraph)
    if not sentences:
        return [paragraph]

    blocks = []
    current_block: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence, tokenizer)

        # If a single sentence itself exceeds max_tokens, hard-split by words
        if sentence_tokens > max_tokens:
            if current_block:
                blocks.append(" ".join(current_block))
                current_block = []
                current_tokens = 0
            # Split the sentence into word groups that fit within max_tokens
            words = sentence.split()
            word_group: List[str] = []
            group_tokens = 0
            for word in words:
                word_tokens = _count_tokens(word, tokenizer)

                # Edge case: single continuous string (Base64, JWT, minified JS) exceeds max_tokens
                if word_tokens > max_tokens:
                    if word_group:
                        blocks.append(" ".join(word_group))
                        word_group = []
                        group_tokens = 0
                    # Hard-slice via tokenizer IDs so the 512-token limit is never breached
                    input_ids = tokenizer.encode(word, add_special_tokens=False)
                    for i in range(0, len(input_ids), max_tokens):
                        sub_ids = input_ids[i:i + max_tokens]
                        blocks.append(tokenizer.decode(sub_ids))
                    continue

                if group_tokens + word_tokens > max_tokens:
                    blocks.append(" ".join(word_group))
                    word_group = [word]
                    group_tokens = word_tokens
                else:
                    word_group.append(word)
                    group_tokens += word_tokens
            if word_group:
                blocks.append(" ".join(word_group))
            continue

        if current_tokens + sentence_tokens > max_tokens and current_block:
            blocks.append(" ".join(current_block))
            current_block = [sentence]
            current_tokens = sentence_tokens
        else:
            current_block.append(sentence)
            current_tokens += sentence_tokens

    if current_block:
        blocks.append(" ".join(current_block))

    return blocks


def chunk_text(
    text: str,
    max_tokens: int = config.CHUNK_SIZE,
    overlap_tokens: int = config.CHUNK_OVERLAP,
    tokenizer=None,
) -> List[Dict]:
    """
    Returns:
        List of dicts: [{"text": str, "chunk_index": int, "token_count": int}]
    """
    text = text.strip()
    if not text:
        return []

    # Load tokenizer if not provided
    if tokenizer is None:
        tokenizer = get_tokenizer()

    # 1. Split text into paragraphs on double newlines
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []

    # 2. Break any oversized paragraphs into sub-blocks
    semantic_blocks: List[str] = []
    for p in raw_paragraphs:
        p_tokens = _count_tokens(p, tokenizer)
        if p_tokens > max_tokens:
            semantic_blocks.extend(
                _split_oversized_paragraph(p, max_tokens, tokenizer)
            )
        else:
            semantic_blocks.append(p)

    # 3. Accumulate blocks into chunks with overlap
    chunks: List[str] = []
    current_chunk_blocks: List[str] = []
    current_tokens = 0

    for block in semantic_blocks:
        block_tokens = _count_tokens(block, tokenizer)

        if current_tokens + block_tokens > max_tokens and current_chunk_blocks:
            # Finalize current chunk
            chunk_str = "\n\n".join(current_chunk_blocks).strip()
            chunks.append(chunk_str)

            # Build overlap from the tail of the previous chunk
            chunk_all_tokens = tokenizer.encode(chunk_str, add_special_tokens=False)
            if len(chunk_all_tokens) > overlap_tokens:
                overlap_text = tokenizer.decode(
                    chunk_all_tokens[-overlap_tokens:], skip_special_tokens=True
                )
                current_chunk_blocks = [overlap_text, block]
                current_tokens = overlap_tokens + block_tokens
            else:
                current_chunk_blocks = [chunk_str, block]
                current_tokens = len(chunk_all_tokens) + block_tokens
        else:
            current_chunk_blocks.append(block)
            current_tokens += block_tokens

    if current_chunk_blocks:
        chunks.append("\n\n".join(current_chunk_blocks).strip())

    # 4. Merge tiny trailing chunk (< 40 tokens) if previous chunk has room
    if len(chunks) > 1:
        last_tokens = _count_tokens(chunks[-1], tokenizer)
        prev_tokens = _count_tokens(chunks[-2], tokenizer)
        if last_tokens < 40 and (prev_tokens + last_tokens <= max_tokens + overlap_tokens):
            tiny_chunk = chunks.pop()
            chunks[-1] = chunks[-1] + "\n\n" + tiny_chunk

    # 5. Format structured output with actual token counts
    result = []
    for idx, c_text in enumerate(chunks):
        result.append({
            "text": c_text,
            "chunk_index": idx,
            "token_count": _count_tokens(c_text, tokenizer),
        })

    return result
