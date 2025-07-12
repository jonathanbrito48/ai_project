import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from setup.integration import integration_run
from utils.llm_interface import LLMInterface
from utils.context_manager import ContextManager
import requests

def pull_ollama_image():
    try:
        response = requests.get("http://ollama:11434/api/tags")
        print(f"Status do modelo 'gemma:2b': {response.status_code}")
        lis_models = response.json().get("models", [])
        print(f"Modelos disponíveis: {lis_models}")
        if not lis_models:
            print("Modelo 'gemma:2b' não encontrado. Iniciando o download...")
            requests.post("http://ollama:11434/api/pull", json={"name": "gemma:2b"})
            print("Download do modelo 'gemma:2b' iniciado.")
        else:
            print("Modelo 'gemma:2b' já está disponível.")
    except requests.exceptions.HTTPError as e:
        if hasattr(e.response, 'status_code') and e.response.status_code == 404:
            print("Modelo 'gemma:2b' não encontrado. Iniciando o download...")
            requests.post("http://ollama:11434/api/pull", json={"name": "gemma:2b"})
            print("Download do modelo 'gemma:2b' iniciado.")
        else:
            print(f"Erro HTTP ao verificar ou baixar o modelo: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Erro ao verificar ou baixar o modelo: {e}")

def main():

    load_dotenv()

    llm_handler = LLMInterface(model_name="gemma:2b", temperature=0.7)

    documents_dir = "data/documents"
    os.makedirs(documents_dir, exist_ok=True)
    
    context_manager = ContextManager(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    context_manager.load_documents(documents_dir)

    print("\n--- Sistema de Perguntas e Respostas Contextualizado ---")
    print("Digite 'sair' para encerrar a conversa.")
    print("-------------------------------------------------------\n")

    system_prompt = SystemMessage(content=(
        "Você é um agente virtual com o objetivo de ajudar o usuário a escolher o melhor carro para compra."
        "Você tem acesso a um contexto que contém informações sobre veículos disponíveis, incluindo marca, modelo, ano, tipo de motor, transmissao, numero de portas, combustivel e preco."
        "Você deve usar essas informações para responder às perguntas do usuário de forma precisa e informativa."
        "Você deve sempre se basear no contexto fornecido e não inventar informações."
        "Você ira sugerir as melhores opções para o perfil do usuário baseado contexto fornecido. Lembre-se de detalhar as especificações mais pertinentes sobre os veículos e explicar o porque das opções serem ideais para ele."
        "Não ofereça apenas uma opção de carro, mas de alternativas de veículos e explique a vantagem deles em relação a preferencia desejada."
        "Você pode ser educado nas saudações e perguntas, mas sempre tentar direcionar a conversa para falar sobre o contexto fornecido. Suas respostas DEVEM ser extraídas e focadas no contexto.\n"
        "Se o contexto não contiver a resposta para a pergunta, ou se as informações forem insuficientes, "
        "Diga que não possui essas informações e que só pode fornecer respostas sobre os contextos fornecidos."
    ))
    
    while True:
        user_query = input("Voce: ")
        if user_query.lower() == 'sair':
            break

        retrieved_context = context_manager.retrieve_context(user_query)

        raw_chat_history = context_manager.get_chat_history_for_llm()

        messages = [system_prompt]

        for msg in raw_chat_history:
            messages.append(msg)

        if retrieved_context:
            messages.append(SystemMessage(content=f"**Informações de Contexto:**\n{retrieved_context}"))
        messages.append(HumanMessage(content=user_query))

        print("\nGerando resposta do LLM...")
        llm_response = llm_handler.generate_response(messages)
        print(f"Assistente: {llm_response}")

        context_manager.add_message_to_history(user_query, is_user=True)
        context_manager.add_message_to_history(llm_response, is_user=False)

    print("\nConversa Encerrada. Até a próxima!")


if __name__ == "__main__":
    pull_ollama_image()
    integration_run()
    main()
