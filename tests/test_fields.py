"""Regras públicas de extração com dados exclusivamente sintéticos."""

import pytest

from openframe_ocr.fields import extract_fields


def pairs(text):
    return [(field.label, field.value) for field in extract_fields(text)]


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("Nome", "Pessoa Exemplo"), ("Nome completo", "Ana Exemplo"),
        ("CPF", "000.000.000-00"), ("RG", "00.000.000-X"),
        ("Data de nascimento", "01/01/2000"), ("Nascimento", "01/01/2000"),
        ("Número do documento", "TESTE-000"), ("Número da CNH", "00000000000"),
        ("Validade", "01/01/2030"), ("Data de emissão", "01/01/2025"),
        ("Endereço", "Rua de Teste, 123"), ("Município", "São Paulo"),
        ("Estado", "São Paulo"), ("UF", "SP"), ("Órgão emissor", "Órgão Exemplo"),
        ("Departamento", "Tecnologia"), ("Responsável", "Maria de Souza"),
        ("Protocolo", "ABC-123"),
    ],
)
def test_explicit_common_and_generic_fields(label, value):
    assert pairs(f"{label}: {value}") == [(label, value)]


@pytest.mark.parametrize("label", ["Nome", "nome", "NOME", "Nome completo", "CPF", "ÓRGÃO EMISSOR"])
def test_common_labels_with_contiguous_value_on_next_line(label):
    assert pairs(f"{label}\nExemplo sintético") == [(label, "Exemplo sintético")]
    assert pairs(f"{label}:\nExemplo sintético") == [(label, "Exemplo sintético")]


def test_case_whitespace_and_unicode_preserve_value_contents():
    text = " \t nOmE   completo  :  JoãO  d'Ávila — 東京  \r\n\tmunicípio:\tSão Luís\t"
    assert pairs(text) == [("nOmE completo", "JoãO  d'Ávila — 東京"), ("município", "São Luís")]
    assert pairs("O\u0301rga\u0303o emissor\nÓrgão de teste") == [("O\u0301rga\u0303o emissor", "Órgão de teste")]


def test_explicit_value_may_contain_colons_without_rewriting():
    assert pairs("Horário: 10:30\nObservação: preservar: este valor") == [
        ("Horário", "10:30"), ("Observação", "preservar: este valor")
    ]


@pytest.mark.parametrize(
    "text",
    [
        "000.000.000-00\n00.000.000-X\n01/01/2000",
        "Texto comum sem campos.", "Este documento contém: informação.",
        "A data é: amanhã", "Leia: as instruções completas.",
        "Nome:\n\nPessoa Exemplo", "Nome\nCPF", "Nome:\nCPF:",
        "Departamento\nTecnologia", "Nome\nUma frase muito longa termina aqui.",
        "Nome:   ", "CPF", "https://exemplo.test", "10:30", ":::valor",
        "Nome\nDepartamento\nTecnologia", "Nome:\nResponsável\nMaria de Souza",
        "Rótulo com mais de cinco palavras diferentes: valor",
    ],
)
def test_ambiguous_unlabeled_and_missing_information_is_not_forced(text):
    assert pairs(text) == []


def test_neighboring_labels_are_not_paired_together():
    assert pairs("Nome\nCPF:\n000.000.000-00\nRG:\nProtocolo: TESTE-123") == [
        ("CPF", "000.000.000-00"), ("Protocolo", "TESTE-123")
    ]


def test_repeated_labels_remain_in_order_with_independent_identity():
    fields = extract_fields("Responsável: Ana Exemplo\nResponsável: Bia Exemplo")
    assert [(field.label, field.value) for field in fields] == [
        ("Responsável", "Ana Exemplo"), ("Responsável", "Bia Exemplo")
    ]
    assert fields[0].id != fields[1].id


@pytest.mark.parametrize("text", ["", " \t\n", None, 42, b"CPF: 000", [], "\x00\ufffd\ud800"])
def test_empty_invalid_or_unrecognized_input_is_safe(text, caplog):
    assert extract_fields(text) == []
    assert caplog.text == ""


@pytest.mark.parametrize(
    ("text", "label", "value"),
    [
        ("Protocolo: MANUAL-001", "Protocolo", "MANUAL-001"),
        ("Protocolo\nMANUAL-001", "Protocolo", "MANUAL-001"),
        ("Protocolo:\nMANUAL-001", "Protocolo", "MANUAL-001"),
        ("  pRoToCoLo  \n  mAnUaL-001  ", "pRoToCoLo", "mAnUaL-001"),
        ("PROTOCOLO\r\nABC-123/2026", "PROTOCOLO", "ABC-123/2026"),
        ("Protocolo\nPROTO_001.2026", "Protocolo", "PROTO_001.2026"),
        ("Protocolo\nÇÃ-001", "Protocolo", "ÇÃ-001"),
        ("Protocolo\n000001", "Protocolo", "000001"),
        ("Protocolo: Texto livre preservado", "Protocolo", "Texto livre preservado"),
    ],
)
def test_protocol_on_one_or_two_lines_preserves_original_value(text, label, value):
    assert pairs(text) == [(label, value)]


@pytest.mark.parametrize(
    "text",
    [
        "Protocolo", "Protocolo:", "Protocolo\n", "Protocolo:\n   ",
        "Protocolo\n\nMANUAL-001", "Protocolo\nCPF", "Protocolo:\nNome:",
        "Protocolo\nDepartamento", "Protocolo\nRELATÓRIO DE TESTE",
        "Protocolo\nRESUMO", "Protocolo\nDocumento 2026",
        "Protocolo\nO prazo termina em 2026", "Protocolo\nInformações abaixo.",
        "Relatório do protocolo\nMANUAL-001", "O protocolo\nMANUAL-001",
        ":\nMANUAL-001", "\nMANUAL-001", "MANUAL-001\n2026",
    ],
)
def test_protocol_missing_ambiguous_or_unlabeled_values_are_not_extracted(text):
    assert pairs(text) == []


def test_protocol_does_not_consume_neighboring_labels_or_replace_generic_fields():
    assert pairs(
        "Nome:\nProtocolo\nMANUAL-001\nProtocolo:\nCPF: 000.000.000-00\n"
        "Departamento: Tecnologia\nProtocolo: OUTRO-002"
    ) == [
        ("Protocolo", "MANUAL-001"), ("CPF", "000.000.000-00"),
        ("Departamento", "Tecnologia"), ("Protocolo", "OUTRO-002"),
    ]
