"""Extração conservadora de pares explícitos, sem dependências de interface."""

from dataclasses import dataclass, field
import unicodedata
from uuid import uuid4


@dataclass
class StructuredField:
    """Um campo independente, inclusive quando há rótulos repetidos."""

    label: str
    value: str
    id: str = field(default_factory=lambda: uuid4().hex)


def _normalize(label: str) -> str:
    return " ".join(
        "".join(
            character
            for character in unicodedata.normalize("NFD", label.casefold())
            if unicodedata.category(character) != "Mn"
        ).split()
    )


_COMMON_LABELS = {
    "nome", "nome completo", "cpf", "rg", "data de nascimento", "nascimento",
    "numero do documento", "numero da cnh", "validade", "data de emissao",
    "endereco", "municipio", "estado", "uf", "orgao emissor", "protocolo",
}
_LABEL_BOUNDARIES = _COMMON_LABELS | {"departamento", "responsavel"}
_SENTENCE_STARTS = {
    "o", "a", "os", "as", "eu", "voce", "ele", "ela", "nos", "isso", "isto",
    "este", "esta", "estes", "estas", "leia", "consulte", "acesse", "verifique",
    "confira", "observe", "veja", "use",
}
_SENTENCE_VERBS = {"é", "são", "foi", "será", "deve", "precisa", "permite"}


def _is_label(label: str) -> bool:
    normalized = _normalize(label)
    if normalized in _COMMON_LABELS:
        return True
    words = normalized.split()
    return (
        0 < len(label) <= 48
        and 0 < len(words) <= 5
        and words[0] not in _SENTENCE_STARTS
        and not (_SENTENCE_VERBS & set(label.casefold().split()))
        and normalized not in {"http", "https", "ftp", "mailto", "tel"}
        and any(character.isalpha() for character in label)
        and all(
            character.isalnum()
            or character.isspace()
            or unicodedata.category(character).startswith("M")
            or character in "/-º°"
            for character in label
        )
    )


def _is_following_value(line: str, label: str) -> bool:
    """Só aceita uma linha contígua curta, sem outro rótulo ou frase evidente."""
    normalized = _normalize(line)
    words = normalized.split()
    unambiguous = (
        bool(words)
        and len(line) <= 160
        and len(words) <= 12
        and ":" not in line
        and normalized not in _LABEL_BOUNDARIES
        and words[0] not in _SENTENCE_STARTS
        and not line.endswith((".", "!", "?", ";"))
    )
    if not unambiguous or _normalize(label) != "protocolo":
        return unambiguous
    # Na ausência de dois-pontos, exige um identificador, em vez de texto livre.
    return any(character.isdigit() for character in line) and all(
        character.isalnum()
        or unicodedata.category(character).startswith("M")
        or character in "-_/."
        for character in line
    )


def extract_fields(text: str) -> list[StructuredField]:
    """Preserva valores e ordem; ignora linhas ambíguas e números sem rótulo."""
    if not isinstance(text, str) or not text.strip():
        return []
    lines = text.splitlines()
    result = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        label, separator, value = line.partition(":")
        label = " ".join(label.split())
        value = value.strip()
        if separator and _is_label(label) and value:
            result.append(StructuredField(label, value))
        elif (
            _normalize(label) in _COMMON_LABELS
            and not value
            and index + 1 < len(lines)
            and _is_following_value(lines[index + 1].strip(), label)
        ):
            result.append(StructuredField(label, lines[index + 1].strip()))
            index += 1
        index += 1
    return result
