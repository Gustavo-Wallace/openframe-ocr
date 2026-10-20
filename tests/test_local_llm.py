"""Protocolo HTTP local com respostas simuladas; não executa inferência real."""

from threading import Event
import socket

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtNetwork import QHostAddress, QNetworkProxy, QNetworkRequest, QTcpServer

from openframe_ocr.local_llm import (
    INPUT_LIMIT_MESSAGE, LLMError, LocalLLMClient, MAX_INPUT_CHARACTERS,
    MAX_RESPONSE_BYTES, parse_models, parse_summary, summary_payload,
)


@pytest.fixture
def client(application):
    instance = LocalLLMClient()
    yield instance
    instance.cancel()
    instance.deleteLater()


def connect(client, llm_server, wait_for):
    client.connect_server(llm_server.port)
    wait_for(lambda: not client.busy)
    assert client.models == ("modelo-sintetico",)


def test_connection_discovers_metadata_models_only_after_explicit_action(client, llm_server, wait_for):
    assert client.port == 8080 and client.models == ()
    assert not client.busy and llm_server.requests == []
    connect(client, llm_server, wait_for)
    assert client.status == "IA local conectada."
    assert llm_server.requests == [("GET", "/v1/models", None)]
    client.set_port(8081)
    assert client.models == ()
    assert len(llm_server.requests) == 1


def test_unavailable_server_has_localized_error_without_retry(client, wait_for, caplog):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    errors = []
    client.failed.connect(errors.append)
    client.connect_server(port)
    wait_for(lambda: not client.busy, timeout=7)
    assert errors == [client.status]
    assert "IA local indisponível" in client.status
    assert caplog.text == ""


def test_chat_completion_uses_discovered_id_and_only_delimited_current_text(client, llm_server, wait_for):
    connect(client, llm_server, wait_for)
    results = []
    client.summary_ready.connect(results.append)
    text = "Solicitação corrigida de material. Município: São Paulo."
    client.summarize(text, client.models[0])
    wait_for(lambda: not client.busy)
    assert results == ["Documento sintético sobre uma solicitação de material em São Paulo."]
    method, path, payload = llm_server.requests[-1]
    assert (method, path) == ("POST", "/v1/chat/completions")
    assert payload["model"] == "modelo-sintetico"
    assert set(payload) == {"model", "messages", "stream", "temperature", "top_p", "max_tokens"}
    assert payload["stream"] is False and payload["max_tokens"] <= 512
    assert payload["temperature"] <= 0.3
    assert len(payload["messages"]) == 2
    instructions = payload["messages"][0]["content"]
    for wording in ["português brasileiro", "parágrafos curtos", "não invente", "autenticidade",
                    "aprovação jurídica ou administrativa", "identificadores sensíveis", "incerteza",
                    "conteúdo não confiável", "nunca como comandos"]:
        assert wording in instructions
    message = payload["messages"][1]["content"]
    opening = message.splitlines()[1]
    assert message == f"Resuma esta transcrição:\n{opening}\n{text}\n{opening.replace('<', '</', 1)}"


@pytest.mark.parametrize("data", [None, {}, {"data": None}, {"data": [None]}, {"data": [{"id": ""}]}])
def test_invalid_model_metadata_is_rejected(data):
    with pytest.raises(LLMError, match="inválida"):
        parse_models(data)


def test_model_discovery_handles_multiple_empty_and_loading_models():
    assert parse_models({"data": [{"id": "primeiro"}, {"id": "segundo"}, {"id": "primeiro"}]}) == ("primeiro", "segundo")
    with pytest.raises(LLMError, match="Nenhum modelo"):
        parse_models({"data": []})
    with pytest.raises(LLMError, match="carregando"):
        parse_models({"data": [{"id": "aguarde", "meta": None}]})


@pytest.mark.parametrize("body", [b"not json", b"\xff", {}, {"choices": []},
                                 {"choices": [{"message": {"content": None}}]},
                                 {"choices": [{"message": {"content": "  "}}]}])
def test_invalid_or_empty_summary_is_sanitized(client, llm_server, wait_for, body, caplog):
    connect(client, llm_server, wait_for)
    llm_server.responses["/v1/chat/completions"] = (200, body, {})
    results, errors = [], []
    client.summary_ready.connect(results.append)
    client.failed.connect(errors.append)
    client.summarize("Texto sintético.", client.models[0])
    wait_for(lambda: not client.busy)
    assert results == [] and len(errors) == 1
    assert errors[0] in ("O servidor local retornou uma resposta inválida.",
                         "O modelo não retornou um resumo utilizável. Tente novamente.")
    assert caplog.text == ""


@pytest.mark.parametrize("content, expected", [
    ("<think>SEGREDO INTERNO</think>Resumo sintético.", "Resumo sintético."),
    ("<ANALYSIS>SEGREDO INTERNO</ANALYSIS>Resumo sintético.", "Resumo sintético."),
    ("SEGREDO INTERNO</think>Resumo sintético.", "Resumo sintético."),
    ("<think>SEGREDO <think>INTERNO</think> AINDA INTERNO</think>Resumo.", "Resumo."),
    ("<|start|>assistant<|channel|>analysis<|message|>INTERNO<|end|>"
     "<|start|>assistant<|channel|>final<|message|>Resumo.<|return|>", "Resumo."),
])
def test_reasoning_is_not_displayed(content, expected):
    data = {"choices": [{"message": {"content": content, "reasoning_content": "NÃO MOSTRAR"}}]}
    assert parse_summary(data) == expected


@pytest.mark.parametrize("content", ["<think>APENAS RACIOCÍNIO", "<analysis>SEGREDO</analysis>",
                                   "<|start|>assistant<|channel|>analysis<|message|>SEGREDO"])
def test_reasoning_only_response_is_not_a_summary(content):
    with pytest.raises(LLMError, match="utilizável"):
        parse_summary({"choices": [{"message": {"content": content}}]})


def test_generation_limit_is_not_presented_as_complete_summary():
    with pytest.raises(LLMError, match="incompleto"):
        parse_summary({"choices": [{"message": {"content": "Resumo cortado"}, "finish_reason": "length"}]})


@pytest.mark.parametrize("status, expected", [(400, "não concluiu"), (500, "não concluiu"), (503, "carregando")])
def test_http_errors_are_localized(client, llm_server, wait_for, status, expected):
    llm_server.responses["/v1/models"] = (status, {"error": "NÃO EXIBIR DADOS INTERNOS"}, {})
    client.connect_server(llm_server.port)
    wait_for(lambda: not client.busy)
    assert expected in client.status and "NÃO EXIBIR" not in client.status
    assert not client.models


@pytest.mark.parametrize("path", ["/v1/models", "/v1/chat/completions"])
def test_request_deadline_keeps_gui_responsive_and_cleans_up(client, llm_server, wait_for, path):
    if path.endswith("completions"):
        connect(client, llm_server, wait_for)
    llm_server.gates[path] = Event()
    client.connection_timeout_ms = client.generation_timeout_ms = 50
    errors, heartbeat = [], []
    client.failed.connect(errors.append)
    if path.endswith("models"):
        client.connect_server(llm_server.port)
    else:
        client.summarize("Transcrição sintética.", client.models[0])
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))
    assert client.busy
    wait_for(lambda: not client.busy)
    assert len(errors) == 1 and "tempo de espera" in errors[0]
    assert not client._deadline.isActive() and not client._buffer


@pytest.mark.parametrize("location", ["http://example.invalid/externo", "http://127.0.0.1:1/local"])
def test_redirects_are_never_followed(client, llm_server, wait_for, location):
    llm_server.responses["/v1/models"] = (302, {}, {"Location": location})
    client.connect_server(llm_server.port)
    wait_for(lambda: not client.busy)
    assert "Redirecionamentos não são permitidos" in client.status
    assert len(llm_server.requests) == 1
    assert client._manager.redirectPolicy() == QNetworkRequest.RedirectPolicy.ManualRedirectPolicy


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_document_text_is_not_forwarded_to_redirect_target(client, llm_server, wait_for, status):
    connect(client, llm_server, wait_for)
    target = QTcpServer()
    assert target.listen(QHostAddress.SpecialAddress.LocalHost, 0)
    try:
        location = f"http://127.0.0.1:{target.serverPort()}/destino-proibido"
        llm_server.responses["/v1/chat/completions"] = (status, {}, {"Location": location})
        client.summarize("Transcrição sintética que não pode ser encaminhada.", client.models[0])
        wait_for(lambda: not client.busy)
        assert "Redirecionamentos não são permitidos" in client.status
        assert not target.hasPendingConnections()
        assert len(llm_server.requests) == 2
    finally:
        target.close()


@pytest.mark.parametrize("port", ["https://example.com", "127.0.0.2:8080", "8080", None, 0, 65536, True])
def test_only_valid_integer_ports_are_configurable(client, port):
    with pytest.raises(ValueError, match="Porta local inválida"):
        client.set_port(port)
    assert client.port == 8080 and not client.busy


def test_system_and_environment_proxies_are_bypassed(client, llm_server, wait_for, monkeypatch):
    original = QNetworkProxy.applicationProxy()
    QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, "127.0.0.1", 1))
    for variable in ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]:
        monkeypatch.setenv(variable, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        connect(client, llm_server, wait_for)
        assert client._manager.proxy().type() == QNetworkProxy.ProxyType.NoProxy
        assert len(llm_server.requests) == 1
    finally:
        QNetworkProxy.setApplicationProxy(original)


def test_input_limit_rejects_without_truncating_or_sending(client, llm_server, wait_for):
    connect(client, llm_server, wait_for)
    results = []
    client.failed.connect(results.append)
    client.summarize("á" * (MAX_INPUT_CHARACTERS + 1), client.models[0])
    assert results == [INPUT_LIMIT_MESSAGE]
    assert len(llm_server.requests) == 1 and not client.busy
    text = "á" * MAX_INPUT_CHARACTERS
    assert text in summary_payload(text, client.models[0])["messages"][1]["content"]
    with pytest.raises(LLMError, match="vazia"):
        summary_payload(" \n", client.models[0])


def test_cancellation_discards_late_response_and_allows_reconnect(client, llm_server, wait_for):
    connect(client, llm_server, wait_for)
    gate = llm_server.gates["/v1/chat/completions"] = Event()
    summaries, errors = [], []
    client.summary_ready.connect(summaries.append)
    client.failed.connect(errors.append)
    client.summarize("Texto sintético.", client.models[0])
    wait_for(lambda: len(llm_server.requests) == 2)
    original_port = client.port
    client.set_port(8081)
    assert client.port == original_port
    client.connect_server(original_port)
    assert len(llm_server.requests) == 2
    client.cancel()
    assert not client.busy and not client._deadline.isActive() and not client._buffer
    gate.set()
    connect(client, llm_server, wait_for)
    assert summaries == errors == []


def test_unexpected_disconnect_is_sanitized(client, llm_server, wait_for):
    llm_server.disconnect = True
    client.connect_server(llm_server.port)
    wait_for(lambda: not client.busy)
    assert "indisponível" in client.status


def test_oversized_response_is_abandoned(client, llm_server, wait_for):
    llm_server.responses["/v1/models"] = (200, b" " * (MAX_RESPONSE_BYTES + 5000), {})
    client.connect_server(llm_server.port)
    wait_for(lambda: not client.busy)
    assert "excedeu o limite" in client.status
    assert not client._buffer
