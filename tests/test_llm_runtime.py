"""Processos reais de curta duração com HTTP simulado, sem llama.cpp ou inferência."""

from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QImage
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QTcpServer, QHostAddress
from PySide6.QtWidgets import QListWidgetItem, QSpinBox, QComboBox

from openframe_ocr import llm_runtime, main_window
from openframe_ocr.document import Document
from openframe_ocr.llm_runtime import LLMRuntime, RuntimeConfig, RuntimeState
from openframe_ocr.local_llm import LocalLLMClient, MAX_INPUT_CHARACTERS


@pytest.fixture
def rig(application, runtime_bundle, wait_for):
    client = LocalLLMClient()
    runtime = LLMRuntime(client, runtime_bundle.config, startup_timeout_ms=3000,
                         poll_interval_ms=25, shutdown_timeout_ms=150,
                         process_factory=runtime_bundle.process_factory)
    state = SimpleNamespace(client=client, runtime=runtime, bundle=runtime_bundle, errors=[])
    runtime.failed.connect(state.errors.append)
    yield state
    runtime.stop()
    wait_for(lambda: not runtime.active, timeout=5)
    assert len(runtime_bundle.exited) == len([item for item in runtime_bundle.instances if item.pid])
    runtime.deleteLater()
    client.deleteLater()


def start(rig, wait_for):
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state in (RuntimeState.READY, RuntimeState.ERROR), timeout=5)
    assert rig.runtime.state == RuntimeState.READY, rig.runtime.status


@pytest.mark.parametrize("missing", ["all", "executable", "model"])
def test_missing_runtime_or_model_never_starts_or_accesses_network(application, runtime_bundle, monkeypatch, missing):
    config = RuntimeConfig()
    if missing == "executable":
        config = RuntimeConfig(runtime_bundle.config.executable.with_name("missing.exe"), runtime_bundle.config.model)
    elif missing == "model":
        config = RuntimeConfig(runtime_bundle.config.executable, runtime_bundle.config.model.with_name("missing.gguf"))
    monkeypatch.setattr(QNetworkAccessManager, "get", lambda *args: pytest.fail("Sem rede sem instalação."))
    client = LocalLLMClient()
    runtime = LLMRuntime(client, config)
    runtime.ensure_ready()
    assert runtime.state == RuntimeState.UNAVAILABLE and not runtime.active
    assert runtime_bundle.instances == []
    runtime.deleteLater()
    client.deleteLater()


@pytest.mark.parametrize("kind", ["relative", "directory", "script", "renamed_script", "model_directory", "model_extension", "model_header"])
def test_invalid_internal_paths_are_rejected_without_execution(tmp_path, runtime_bundle, kind):
    executable, model = runtime_bundle.config.executable, runtime_bundle.config.model
    if kind == "relative":
        executable = Path("llama-server.exe")
    elif kind == "directory":
        executable = tmp_path
    elif kind in ("script", "renamed_script"):
        executable = tmp_path / ("arquivo.py" if kind == "script" else "arquivo.exe")
        executable.write_text("print('never execute')")
    elif kind == "model_directory":
        model = tmp_path
    elif kind == "model_extension":
        model = model.with_suffix(".txt")
        model.write_bytes(b"GGUF synthetic test")
    else:
        model = model.with_name("invalid.gguf")
        model.write_bytes(b"invalid model test")
    with pytest.raises(ValueError, match="ausentes ou inválidos"):
        RuntimeConfig(executable, model).validated()


def test_internal_paths_preserve_spaces_and_unicode(runtime_bundle):
    executable, model = runtime_bundle.config.validated()
    assert executable.is_absolute() and model.is_absolute()
    assert "ação" in str(model) and "modelo sintético" in model.name


def test_started_is_not_ready_and_health_503_is_polled_before_authenticated_models(rig, wait_for):
    rig.bundle.set_options(loading_requests=2)
    rig.runtime.ensure_ready()
    wait_for(lambda: bool(rig.bundle.requests()))
    assert rig.runtime.state == RuntimeState.STARTING
    assert not rig.client.models
    wait_for(lambda: rig.runtime.state == RuntimeState.READY)
    records = rig.bundle.requests()
    assert [item["path"] for item in records] == ["/health"] * 3 + ["/v1/models"] * 2
    assert not records[-2]["authenticated"] and records[-1]["authenticated"]
    assert rig.runtime.model_id.startswith("openframe-")
    assert not rig.runtime._startup.isActive() and not rig.runtime._poll.isActive()


def test_repeated_start_reuses_one_owned_process_and_completions_use_session_auth(rig, wait_for):
    rig.runtime.ensure_ready()
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.READY)
    original = rig.bundle.instances[0]
    pid = original.pid
    rig.runtime.ensure_ready()
    assert len(rig.bundle.instances) == 1 and original.pid == pid
    assert rig.runtime.owns_endpoint()
    results = []
    rig.client.summary_ready.connect(results.append)
    rig.client.summarize("Documento sintético em português.", rig.runtime.model_id)
    wait_for(lambda: bool(results))
    assert results == ["Resumo sintético do documento de teste."]
    assert rig.bundle.requests()[-1] == {"path": "/v1/chat/completions", "authenticated": True}
    args = original.launch_arguments
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--n-gpu-layers") + 1] == "0"
    assert "--offline" in args and "--log-disable" in args
    assert rig.runtime._token not in " ".join(args)
    assert original.processEnvironment().value("LLAMA_API_KEY") == rig.runtime._token


@pytest.mark.parametrize("options", [{"health_body": {}}, {"no_auth": True}, {"model_id": "outro-servidor"}, {"redirect": True}])
def test_invalid_health_authentication_identity_and_redirect_fail_closed(rig, wait_for, options):
    rig.bundle.set_options(**options)
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert rig.errors and not rig.client.models
    assert not any(item["path"].endswith("completions") for item in rig.bundle.requests())
    assert rig.runtime.can_request


def test_cpu_startup_timeout_is_bounded_and_retry_possible(rig, wait_for):
    rig.bundle.set_options(always_loading=True)
    rig.runtime._startup.setInterval(350)
    rig.runtime.ensure_ready()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    wait_for(lambda: bool(heartbeat))
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert "tempo de espera" in rig.runtime.status
    assert not rig.runtime._poll.isActive() and rig.runtime._probe is None
    rig.bundle.set_options()
    rig.runtime._startup.setInterval(3000)
    start(rig, wait_for)
    assert len(rig.bundle.instances) == 2


def test_process_exit_before_readiness_is_detected_and_cleaned(rig, wait_for):
    rig.bundle.set_options(exit_early=True)
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR)
    assert not rig.runtime.active and not rig.client.models
    assert "encerrada inesperadamente" in rig.runtime.status
    assert rig.runtime._probe is None


def test_cancellation_immediately_after_start_does_not_leave_a_child(rig, wait_for):
    rig.runtime.ensure_ready()
    rig.runtime.stop()
    rig.runtime.stop()
    wait_for(lambda: not rig.runtime.active)
    assert rig.runtime.state == RuntimeState.STOPPED
    assert not rig.runtime._startup.isActive() and not rig.runtime._poll.isActive()
    assert rig.runtime._probe is None


def test_hanging_health_request_is_abandoned_on_startup_deadline(rig, wait_for):
    rig.bundle.set_options(health_delay=1)
    rig.runtime._startup.setInterval(350)
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert rig.runtime._probe is None and not rig.runtime._probe_timeout.isActive()
    assert not rig.client.busy


def test_endpoint_ownership_is_rechecked_before_sending_document(rig, wait_for, monkeypatch):
    start(rig, wait_for)
    count = len(rig.bundle.requests())
    monkeypatch.setattr(llm_runtime, "_listener_pid", lambda port: 1)
    rig.client.summarize("Nunca encaminhar este documento a outro processo.", rig.runtime.model_id)
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert len(rig.bundle.requests()) == count
    assert not rig.client.models


def test_ownership_check_failure_never_falls_back_to_unverified_http(rig, wait_for, monkeypatch):
    def denied(port):
        raise OSError("synthetic denial")
    monkeypatch.setattr(llm_runtime, "_listener_pid", denied)
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert rig.bundle.requests() == []
    assert "conexão segura" in rig.runtime.status


def test_native_launch_failure_is_localized(application, runtime_bundle, tmp_path, wait_for):
    executable = tmp_path / "broken native.exe"
    executable.write_bytes(b"MZinvalid native runtime")
    client = LocalLLMClient()
    runtime = LLMRuntime(client, RuntimeConfig(executable, runtime_bundle.config.model))
    try:
        runtime.ensure_ready()
        wait_for(lambda: runtime.state == RuntimeState.ERROR)
        assert not runtime.active and "Não foi possível iniciar" in runtime.status
    finally:
        runtime.stop()
        runtime.deleteLater()
        client.deleteLater()


@pytest.mark.parametrize("graceful", [True, False])
def test_shutdown_only_owned_process_is_async_idempotent_and_clears_secrets(rig, wait_for, graceful):
    rig.bundle.graceful = graceful
    start(rig, wait_for)
    process = rig.bundle.instances[0]
    rig.runtime.stop()
    rig.runtime.stop()
    wait_for(lambda: not rig.runtime.active)
    assert rig.runtime.state == RuntimeState.STOPPED
    assert process.terminated == 1
    assert process.killed == (0 if graceful else 1)
    assert not rig.runtime._token and not rig.client._token and not rig.client.models
    assert not rig.runtime._shutdown.isActive()


def test_port_race_retries_without_connecting_to_or_terminating_unrelated_service(rig, wait_for, monkeypatch, llm_server):
    ports = iter([llm_server.port, llm_runtime._free_port()])
    monkeypatch.setattr(llm_runtime, "_free_port", lambda: next(ports))
    start(rig, wait_for)
    assert llm_server.requests == []
    assert rig.runtime._attempt == 2
    assert len(rig.bundle.instances) == 2
    assert rig.bundle.instances[0].pid in rig.bundle.exited
    # O serviço externo continua ouvindo depois do conflito e da limpeza.
    with socket.create_connection(("127.0.0.1", llm_server.port), timeout=1):
        pass


def test_port_conflicts_have_a_small_bounded_retry_limit(rig, wait_for, monkeypatch, llm_server):
    monkeypatch.setattr(llm_runtime, "_free_port", lambda: llm_server.port)
    rig.runtime.ensure_ready()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert len(rig.bundle.instances) == 3
    assert llm_server.requests == []


def test_proxy_settings_and_parent_agent_settings_are_not_used(rig, wait_for, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "exec_shell_command")
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    monkeypatch.setenv("LLAMA_ARG_HF_REPO", "do-not-download")
    original = QNetworkProxy.applicationProxy()
    QNetworkProxy.setApplicationProxy(QNetworkProxy(QNetworkProxy.ProxyType.HttpProxy, "127.0.0.1", 1))
    try:
        start(rig, wait_for)
        assert rig.runtime._network.proxy().type() == QNetworkProxy.ProxyType.NoProxy
        environment = rig.bundle.instances[0].processEnvironment()
        assert not environment.contains("LLAMA_ARG_TOOLS")
        assert not environment.contains("LLAMA_ARG_HOST")
        assert not environment.contains("LLAMA_ARG_HF_REPO")
    finally:
        QNetworkProxy.setApplicationProxy(original)


def test_unexpected_exit_after_ready_cancels_requests_and_allows_restart(rig, wait_for):
    start(rig, wait_for)
    rig.bundle.set_options(generation_delay=1)
    rig.client.summarize("Documento sintético.", rig.runtime.model_id)
    wait_for(lambda: any(item["path"].endswith("completions") for item in rig.bundle.requests()))
    rig.bundle.instances[0].kill()
    wait_for(lambda: rig.runtime.state == RuntimeState.ERROR and not rig.runtime.active)
    assert not rig.client.busy and not rig.client.models
    rig.bundle.set_options()
    start(rig, wait_for)


@pytest.fixture
def managed_window(application, runtime_bundle, monkeypatch, wait_for):
    def create(client, config, parent):
        return LLMRuntime(client, config, parent, startup_timeout_ms=3000,
                          poll_interval_ms=25, shutdown_timeout_ms=150,
                          process_factory=runtime_bundle.process_factory)
    monkeypatch.setattr(main_window, "LLMRuntime", create)
    window = main_window.MainWindow(runtime_config=runtime_bundle.config)
    image = QImage(640, 480, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    document = Document("Documento sintético", image=image)
    item = QListWidgetItem(document.name)
    item.setData(Qt.ItemDataRole.UserRole, document)
    window.document_list.addItem(item)
    window.document_list.setCurrentItem(item)
    window.text_edit.setPlainText("Solicitação de material para o laboratório em São Paulo.")
    window.show()
    yield window
    window.close()
    wait_for(lambda: not window._runtime.active)
    assert len(runtime_bundle.exited) == len([item for item in runtime_bundle.instances if item.pid])


def test_summary_click_automatically_prepares_runtime_and_reuses_it(managed_window, runtime_bundle, wait_for):
    window = managed_window
    panel = window.assistant_panel
    assert runtime_bundle.instances == []
    panel.toggle.click()
    assert runtime_bundle.instances == []
    assert panel.findChildren(QSpinBox) == panel.findChildren(QComboBox) == []
    panel.summarize_button.click()
    assert "Preparando IA" in panel.status.text()
    wait_for(lambda: bool(window.current_document.summary))
    assert window._runtime.state == RuntimeState.READY
    assert panel.runtime_status.text() == "IA pronta."
    panel.summarize_button.click()
    wait_for(lambda: not window._llm.busy)
    assert len(runtime_bundle.instances) == 1


@pytest.mark.parametrize("during", ["startup", "generation"])
def test_close_during_model_loading_or_generation_releases_child(managed_window, runtime_bundle, wait_for, during):
    window = managed_window
    if during == "startup":
        runtime_bundle.set_options(always_loading=True)
    else:
        runtime_bundle.set_options(generation_delay=1)
    window.assistant_panel.summarize_button.click()
    wait_for(lambda: bool(runtime_bundle.requests()))
    if during == "generation":
        wait_for(lambda: any(item["path"].endswith("completions") for item in runtime_bundle.requests()))
    assert not window.close()
    wait_for(lambda: not window._runtime.active and not window.isVisible())
    assert not window._llm.busy and window._summary_request is None


def test_changes_during_loading_and_oversized_input_never_send_stale_text(managed_window, runtime_bundle, wait_for):
    window = managed_window
    window.text_edit.setPlainText("á" * (MAX_INPUT_CHARACTERS + 1))
    window.assistant_panel.summarize_button.click()
    assert runtime_bundle.instances == []
    assert "4.000" in window.assistant_panel.status.text()
    runtime_bundle.set_options(loading_requests=3)
    window.text_edit.setPlainText("Texto sintético original.")
    window.assistant_panel.summarize_button.click()
    wait_for(lambda: bool(runtime_bundle.requests()))
    window.text_edit.setPlainText("Texto sintético corrigido.")
    wait_for(lambda: window._runtime.state == RuntimeState.READY)
    assert not window.current_document.summary
    assert "Texto alterado durante a preparação" in window.assistant_panel.status.text()
    assert not any(item["path"].endswith("completions") for item in runtime_bundle.requests())
