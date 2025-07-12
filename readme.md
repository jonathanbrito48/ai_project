# Projeto AI Car Recommendation

Este projeto é um sistema de perguntas e respostas contextualizado para auxiliar usuários na escolha do melhor carro para compra, utilizando modelos de linguagem natural e banco de dados de veículos.

## Funcionalidades

- Recomenda veículos com base nas preferências do usuário.
- Utiliza embeddings e busca semântica para contextualizar respostas.
- Integração com modelo LLM via [Ollama](https://ollama.com/).
- Banco de dados de veículos com integração ETL.
- Suporte a documentos externos (txt/pdf) para enriquecer o contexto.

## Estrutura do Projeto

- `main.py`: Entrada principal do sistema.
- `setup/`: Scripts de integração e modelos do banco de dados.
- `utils/`: Utilitários para contexto e interface com LLM.
- `data/documents/`: Documentos para enriquecer o contexto.
- `faiss_index/`: Índice vetorial FAISS para busca semântica.
- `test_mcp.py`: Testes automatizados com pytest.

## Como rodar com Docker

### 1. Build e execução do sistema

Certifique-se de ter o [Docker](https://www.docker.com/) e [Docker Compose](https://docs.docker.com/compose/) instalados.

No terminal, execute:

```sh
docker-compose up --build
```

Isso irá:
- Subir o serviço do Ollama (modelo LLM).
- Subir o serviço da aplicação Python, que inicializa o banco, carrega documentos e inicia o sistema de perguntas e respostas.

A aplicação ficará disponível no terminal do container `app`, pronta para interação.

### 2. Rodando os testes (pytest) via Docker

Para rodar os testes automatizados, utilize o seguinte comando:

```sh
docker-compose run --rm app pytest
```

Isso executará os testes definidos em [`test_mcp.py`](test_mcp.py) no ambiente Docker, garantindo que todas as dependências estejam corretas.

## Como iniciar a interação com o MCP

Atenção: O comando `docker-compose up -d --build` apenas constrói e sobe os containers em modo daemon, mas **não inicia a interação com o MCP**.

Para iniciar o sistema de perguntas e respostas (MCP), utilize:

```sh
docker-compose run app
```

Esse comando executa o container da aplicação e já inicia o MCP, permitindo a interação diretamente pelo terminal.

Se preferir acessar o shell do container para rodar comandos manualmente, utilize:

```sh
docker-compose run --entrypoint bash app
```

E então, dentro do container, execute:

```sh
python main.py
```

---

Dessa forma, o usuário entende que a interação ocorre via `docker-compose run app` e não apenas ao subir os containers.

## Observações

- O banco de dados é inicializado automaticamente e populado via ETL na primeira execução.
- Os documentos para contexto devem ser colocados em `data/documents/` (formatos `.txt` ou `.pdf`).
- O índice vetorial FAISS é salvo em `faiss_index/` e persistido entre execuções.
- Para adicionar novos veículos, atualize o dataset e rode novamente a integração.

## Requisitos

- Docker e Docker Compose
- (Opcional) Python 3.12+ para execução local.
    Obs: Para execução local, é necessário alterar as rotas HTTP de http://ollama:11434 para http://localhost:11434 nos arquivos llm_interface.py e main.py
