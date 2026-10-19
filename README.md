# OpenFrame OCR

Aplicativo desktop open-source em Python para digitalização de documentos e extração local de texto a partir de imagens e webcam.

## Funcionalidades planejadas

- Captura de documentos pela webcam e importação de imagens PNG/JPEG.
- OCR local com texto editável e campos copiáveis individualmente.
- Análise opcional de nitidez, iluminação, contraste e legibilidade.
- Resumos e perguntas sobre documentos utilizando uma LLM local opcional.
- Funcionamento básico sem necessidade de GPU dedicada.

## Requisitos

- Python 3.11 ou superior.
- Ambiente gráfico compatível com Qt.

## Execução a partir do código

Clone o repositório e crie um ambiente virtual:

```sh
git clone https://github.com/Gustavo-Wallace/openframe-ocr.git
cd openframe-ocr
python -m venv .venv
```

Ative o ambiente virtual.

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux ou macOS:

```sh
source .venv/bin/activate
```

Instale e execute a aplicação:

```sh
python -m pip install -e .
python -m openframe_ocr
```

## Testes

Instale as dependências de desenvolvimento e execute os testes:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

---

Desenvolvido por Gustavo Wallace.
