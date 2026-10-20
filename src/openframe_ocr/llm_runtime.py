"""Ciclo de vida do llama-server instalado e pertencente ao aplicativo."""

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import secrets
import socket
import sys

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkRequest

from openframe_ocr.local_llm import LocalLLMClient


class RuntimeState(Enum):
    UNAVAILABLE = "unavailable"
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeConfig:
    """Caminhos confiáveis fornecidos internamente pela instalação, nunca pela UI."""

    executable: Path | None = None
    model: Path | None = None

    def validated(self) -> tuple[Path, Path]:
        try:
            executable, model = Path(self.executable), Path(self.model)
            if not executable.is_absolute() or not model.is_absolute():
                raise ValueError
            if str(executable).startswith("\\\\") or str(model).startswith("\\\\"):
                raise ValueError
            executable = executable.resolve(strict=True)
            model = model.resolve(strict=True)
            if str(executable).startswith("\\\\") or str(model).startswith("\\\\"):
                raise ValueError
            if not executable.is_file() or not model.is_file() or model.suffix.lower() != ".gguf":
                raise ValueError
            with executable.open("rb") as source:
                magic = source.read(4)
            if sys.platform == "win32":
                if executable.suffix.lower() != ".exe" or not magic.startswith(b"MZ"):
                    raise ValueError
            elif magic != b"\x7fELF" or not os.access(executable, os.X_OK):
                raise ValueError
            with model.open("rb") as source:
                if source.read(4) != b"GGUF":
                    raise ValueError
        except (OSError, RuntimeError, TypeError, ValueError):
            raise ValueError("IA não instalada. Os arquivos da IA local estão ausentes ou inválidos.") from None
        return executable, model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        return reservation.getsockname()[1]


def _listener_pid(port: int) -> int | None:
    """Consulta local de posse da porta antes de enviar qualquer HTTP ou segredo."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        size = wintypes.ULONG(0)
        api = ctypes.WinDLL("iphlpapi").GetExtendedTcpTable
        api.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.ULONG), wintypes.BOOL,
                        wintypes.ULONG, wintypes.ULONG, wintypes.ULONG]
        api.restype = wintypes.ULONG
        result = api(None, ctypes.byref(size), False, socket.AF_INET, 3, 0)
        if result not in (0, 122):
            raise OSError("Não foi possível verificar a porta local.")
        buffer = ctypes.create_string_buffer(size.value)
        if api(buffer, ctypes.byref(size), False, socket.AF_INET, 3, 0) != 0:
            raise OSError("Não foi possível verificar a porta local.")
        # MIB_TCPTABLE_OWNER_PID: contagem DWORD seguida por linhas de seis DWORDs.
        values = memoryview(buffer).cast("B")
        count = int.from_bytes(values[:4], "little")
        for index in range(count):
            row = values[4 + index * 24:4 + (index + 1) * 24]
            local_port = socket.ntohs(int.from_bytes(row[8:12], "little") & 0xffff)
            if local_port == port:
                if bytes(row[4:8]) != socket.inet_aton("127.0.0.1"):
                    return -1
                return int.from_bytes(row[20:24], "little")
        return None
    if sys.platform.startswith("linux"):
        inode = None
        for row in Path("/proc/net/tcp").read_text().splitlines()[1:]:
            fields = row.split()
            if fields[3] == "0A" and int(fields[1].split(":")[1], 16) == port:
                if fields[1].split(":")[0] != "0100007F":
                    return -1
                inode = fields[9]
                break
        if inode is None:
            return None
        # Só inspeciona os descritores do próprio filho em owns_endpoint().
        return -int(inode)
    raise OSError("A verificação de porta não está disponível neste sistema.")


class LLMRuntime(QObject):
    changed = Signal()
    ready = Signal()
    failed = Signal(str)

    def __init__(self, client: LocalLLMClient, config: RuntimeConfig | None = None, parent=None,
                 *, startup_timeout_ms=180000, poll_interval_ms=500, shutdown_timeout_ms=3000,
                 process_factory=QProcess):
        super().__init__(parent)
        self.client = client
        self.config = config or RuntimeConfig()
        self.state = RuntimeState.STOPPED if self.available else RuntimeState.UNAVAILABLE
        self.status = "IA local disponível." if self.available else "IA não instalada."
        self.model_id = ""
        self._process = None
        self._process_factory = process_factory
        self._port = 0
        self._token = ""
        self._alias = ""
        self._attempt = 0
        self._after_stop = "stopped"
        self._last_error = ""
        self._probe = None
        self._probe_kind = ""
        self._probe_data = bytearray()
        self._network = QNetworkAccessManager(self)
        self._network.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self._network.setRedirectPolicy(QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        self._poll = QTimer(self)
        self._poll.setSingleShot(True)
        self._poll.setInterval(poll_interval_ms)
        self._poll.timeout.connect(self._check_health)
        self._startup = QTimer(self)
        self._startup.setSingleShot(True)
        self._startup.setInterval(startup_timeout_ms)
        self._startup.timeout.connect(lambda: self._fail("A preparação da IA local excedeu o tempo de espera. Tente novamente."))
        self._shutdown = QTimer(self)
        self._shutdown.setSingleShot(True)
        self._shutdown.setInterval(shutdown_timeout_ms)
        self._shutdown.timeout.connect(self._kill)
        self._probe_timeout = QTimer(self)
        self._probe_timeout.setSingleShot(True)
        self._probe_timeout.setInterval(3000)
        self._probe_timeout.timeout.connect(self._probe_timed_out)
        client.models_ready.connect(self._models_ready)
        client.failed.connect(self._client_failed)
        client.connection_lost.connect(self._connection_lost)

    @property
    def available(self) -> bool:
        try:
            self.config.validated()
            return True
        except ValueError:
            return False

    @property
    def active(self) -> bool:
        return self._process is not None

    @property
    def can_request(self) -> bool:
        return self.available and self.state in (RuntimeState.STOPPED, RuntimeState.READY, RuntimeState.ERROR) and (
            self.state == RuntimeState.READY or not self.active
        )

    def owns_endpoint(self) -> bool:
        process = self._process
        if process is None or process.state() != QProcess.ProcessState.Running:
            return False
        owner = _listener_pid(self._port)
        if sys.platform.startswith("linux") and owner is not None and owner < 0:
            target = f"socket:[{-owner}]"
            return any(os.readlink(fd) == target for fd in Path(f"/proc/{process.processId()}/fd").iterdir())
        return owner == process.processId()

    def ensure_ready(self) -> None:
        if self.state in (RuntimeState.STARTING, RuntimeState.STOPPING):
            return
        if self.state == RuntimeState.READY:
            try:
                if self.owns_endpoint():
                    self.ready.emit()
                    return
            except OSError:
                pass
            self._fail("A IA local ficou indisponível. Tente novamente.")
            return
        if self.active:
            return
        try:
            self.config.validated()
        except ValueError as error:
            self._set_state(RuntimeState.UNAVAILABLE, str(error))
            self.failed.emit(str(error))
            return
        self._attempt = 0
        self._startup.start()
        self._launch()

    def _set_state(self, state, message) -> None:
        self.state, self.status = state, message
        self.changed.emit()

    def _launch(self) -> None:
        try:
            executable, model = self.config.validated()
            self._port = _free_port()
        except (ValueError, OSError):
            self._fail("Não foi possível preparar os arquivos ou a conexão da IA local.")
            return
        self._attempt += 1
        self._token = secrets.token_urlsafe(32)
        self._alias = "openframe-" + secrets.token_hex(16)
        self.model_id = ""
        self._after_stop = ""
        process = self._process_factory(self)
        self._process = process
        environment = QProcessEnvironment.systemEnvironment()
        for key in environment.keys():
            if key.startswith("LLAMA_") or key.upper() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                environment.remove(key)
        environment.insert("LLAMA_API_KEY", self._token)
        environment.insert("HF_HUB_OFFLINE", "1")
        process.setProcessEnvironment(environment)
        process.setProgram(str(executable))
        process.setArguments([
            "--model", str(model), "--host", "127.0.0.1", "--port", str(self._port),
            "--alias", self._alias, "--n-gpu-layers", "0", "--ctx-size", "8192",
            "--parallel", "1", "--offline", "--no-webui", "--log-disable",
        ])
        process.setWorkingDirectory(str(executable.parent))
        process.setStandardOutputFile(QProcess.nullDevice())
        process.setStandardErrorFile(QProcess.nullDevice())
        process.started.connect(lambda: self._started(process))
        process.finished.connect(lambda code, status: self._exited(process))
        process.errorOccurred.connect(lambda error: self._process_error(process, error))
        self.client.configure_managed(self._port, self._token, self._guard)
        self._set_state(RuntimeState.STARTING, "Preparando IA local...")
        process.start()

    def _guard(self) -> bool:
        try:
            return self.state in (RuntimeState.STARTING, RuntimeState.READY) and self.owns_endpoint()
        except OSError:
            return False

    def _started(self, process) -> None:
        if process is not self._process:
            return
        if self.state == RuntimeState.STOPPING:
            process.terminate()
            self._shutdown.start()
            return
        self._set_state(RuntimeState.STARTING, "Carregando modelo...")
        self._poll.start()

    def _check_health(self) -> None:
        if self.state != RuntimeState.STARTING or self._probe is not None:
            return
        try:
            if not self.owns_endpoint():
                if _listener_pid(self._port) is not None:
                    self._retry_port()
                else:
                    self._poll.start()
                return
        except OSError:
            self._fail("Não foi possível verificar a conexão segura da IA local.")
            return
        self._request_probe("health", "/health")

    def _request_probe(self, kind, path) -> None:
        if not self._guard():
            self._fail("A conexão da IA local não pôde ser confirmada.")
            return
        request = QNetworkRequest(QUrl(f"http://127.0.0.1:{self._port}{path}"))
        request.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute,
                             QNetworkRequest.RedirectPolicy.ManualRedirectPolicy)
        request.setTransferTimeout(3000)
        self._probe_kind = kind
        self._probe_data.clear()
        reply = self._network.get(request)
        self._probe = reply
        reply.setReadBufferSize(8193)
        reply.readyRead.connect(lambda: self._read_probe(reply))
        reply.finished.connect(lambda: self._probe_finished(reply))
        self._probe_timeout.start()

    def _read_probe(self, reply) -> None:
        if reply is not self._probe or not reply.isReadable():
            return
        self._probe_data.extend(bytes(reply.readAll()))
        if len(self._probe_data) > 8192:
            self._fail("A IA local retornou uma resposta de preparação inválida.")

    def _probe_finished(self, reply) -> None:
        if reply is not self._probe or self.state != RuntimeState.STARTING:
            return
        self._read_probe(reply)
        if reply is not self._probe:
            return
        self._probe = None
        self._probe_timeout.stop()
        code = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        kind = self._probe_kind
        data = bytes(self._probe_data)
        self._probe_data.clear()
        reply.deleteLater()
        if code == 503:
            self._poll.start()
        elif kind == "auth" and code == 401:
            self.client.connect_server(self._port)
        elif kind == "health" and code == 200:
            try:
                if json.loads(data).get("status") != "ok":
                    raise ValueError
            except (ValueError, AttributeError, UnicodeError, RecursionError):
                self._fail("A IA local retornou uma resposta de preparação inválida.")
                return
            self._request_probe("auth", "/v1/models")
        else:
            self._fail("Não foi possível confirmar a preparação e a autenticação da IA local.")

    def _models_ready(self) -> None:
        if self.state != RuntimeState.STARTING:
            return
        if self.client.models != (self._alias,) or not self._guard():
            self._fail("Não foi possível confirmar a identidade da IA local.")
            return
        self.model_id = self.client.models[0]
        self._startup.stop()
        self._set_state(RuntimeState.READY, "IA pronta.")
        self.ready.emit()

    def _client_failed(self, message) -> None:
        if self.state == RuntimeState.STARTING:
            self._fail("Não foi possível preparar o modelo da IA local. Tente novamente.")

    def _connection_lost(self) -> None:
        if self.state == RuntimeState.READY:
            self._fail("A IA local ficou indisponível. Tente novamente.")

    def _probe_timed_out(self) -> None:
        self._cancel_probe()
        if self.state == RuntimeState.STARTING:
            self._poll.start()

    def _cancel_probe(self) -> None:
        reply, self._probe = self._probe, None
        self._probe_timeout.stop()
        self._probe_data.clear()
        if reply is not None:
            reply.abort()
            reply.deleteLater()

    def _retry_port(self) -> None:
        if self._attempt >= 3:
            self._fail("Não foi possível reservar uma conexão para a IA local. Tente novamente.")
            return
        self._begin_stop("retry")

    def _fail(self, message) -> None:
        self._last_error = message
        self._startup.stop()
        self._begin_stop("error")
        self.failed.emit(message)

    def stop(self) -> None:
        if self.state == RuntimeState.STOPPING:
            self._after_stop = "stopped"
            self._startup.stop()
            return
        self._startup.stop()
        self._begin_stop("stopped")

    def _begin_stop(self, after) -> None:
        self._after_stop = after
        self._poll.stop()
        self._cancel_probe()
        self.client.cancel()
        self.client.clear_managed()
        self.model_id = ""
        process = self._process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._clean_process()
            return
        self._set_state(RuntimeState.STOPPING, "Finalizando IA local...")
        self._shutdown.start()
        if process.state() == QProcess.ProcessState.Running:
            process.terminate()

    def _kill(self) -> None:
        if self._process is not None and self.state == RuntimeState.STOPPING:
            self._process.kill()

    def _process_error(self, process, error) -> None:
        if process is self._process and error == QProcess.ProcessError.FailedToStart:
            self._fail("Não foi possível iniciar a IA local. Verifique a instalação da IA.")

    def _exited(self, process) -> None:
        if process is not self._process:
            return
        if self.state != RuntimeState.STOPPING:
            try:
                conflict = self.state == RuntimeState.STARTING and _listener_pid(self._port) is not None
            except OSError:
                conflict = False
            if conflict and self._attempt < 3:
                self._after_stop = "retry"
            else:
                self._last_error = (
                    "Não foi possível reservar uma conexão para a IA local. Tente novamente."
                    if conflict else "A IA local foi encerrada inesperadamente. Tente novamente."
                )
                self._after_stop = "error"
                self._startup.stop()
                self.failed.emit(self._last_error)
        self._clean_process()

    def _clean_process(self) -> None:
        self._shutdown.stop()
        self._poll.stop()
        self._cancel_probe()
        self.client.cancel()
        self.client.clear_managed()
        self._network.clearConnectionCache()
        process, self._process = self._process, None
        if process is not None:
            process.setProcessEnvironment(QProcessEnvironment())
            process.deleteLater()
        self._token = self._alias = self.model_id = ""
        action = self._after_stop
        self._after_stop = ""
        if action == "retry":
            self._set_state(RuntimeState.STARTING, "Preparando IA local...")
            QTimer.singleShot(0, self._retry_if_starting)
        elif action == "error":
            self._set_state(RuntimeState.ERROR, self._last_error)
        else:
            state = RuntimeState.STOPPED if self.available else RuntimeState.UNAVAILABLE
            self._set_state(state, "IA local disponível." if self.available else "IA não instalada.")

    def _retry_if_starting(self):
        if self.state == RuntimeState.STARTING and self._startup.isActive() and not self.active:
            self._launch()
