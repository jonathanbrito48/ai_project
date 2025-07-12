import requests
import pytest
from utils.context_manager import ContextManager
from utils.llm_interface import LLMInterface
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document


# Define the system prompt
system_prompt = SystemMessage(content=(
    "Você é um agente virtual com o objetivo de ajudar o usuário a escolher o melhor carro para compra."
    "Quando tiver informações suficientes sobre as preferencias do usuário"
    "Você ira sugerir as melhores opções para o perfil do usuário baseado contexto fornecido (um dicionario de carros). Lembre-se de detalhar as especificações mais pertinentes sobre os veículos e explicar o porque das opções serem ideais para ele."
    "Não ofereça apenas uma opção de carro, mas de alternativas de veículos e explique a vantagem deles em relação a preferencia desejada."
    "Você pode ser educado nas saudações e perguntas, mas sempre tentar direcionar a conversa para falar sobre o contexto fornecido. Suas respostas DEVEM ser extraídas e focadas no contexto.\n"
    "Se o contexto não contiver a resposta para a pergunta, ou se as informações forem insuficientes, "
    "Diga que não possui essas informações e que só pode fornecer respostas sobre os contextos fornecidos."
))

# Define the questions and expected responses
questions = {
    "q1_in_context": {
        "question": "Me ajude a escolher um carro no tipo SUV com orçamento de até 150 mil, 4 portas e economico?",
        "expected_response": ["SUV", "veículos", "carro", "melhor", "orçamento", "portas", "econômico","tabela","sugerimos","combustivel"]
    },
    "q2_out_of_context": {
        "question": "Qual é o maior animal do mundo?",
        "expected_response": ["Não", "posso", "informações", "sobre","ajudar","sugerir","pergunta","fornecer"]
    }
}


# Test function to validate the response generation
def test_response_generation():
    llm_handler = LLMInterface(model_name="gemma:2b", temperature=0.1)

    context_manager = ContextManager(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")
    context_manager.load_documents('data/documents')

    result = []

    for q in questions:
        print(f"----\nTestando pergunta de {q}: {questions[q]['question']}\n-----")

        retrieved_context = context_manager.retrieve_context(questions[q]['question'])

        messages = [system_prompt]

        if retrieved_context:
            messages.append(SystemMessage(content=f"**Informações de Contexto:**\n{retrieved_context}"))
        messages.append(HumanMessage(content=questions[q]['question']))

        response = llm_handler.generate_response(messages)

        content_response = [word.replace(".","").replace(",","").replace("'","").replace("*","").lower() for word in response.split(" ")]

        print(f"Resposta gerada: {response.split("\n")}\n-----")

        response_assertions = 0
        palavras_encontradas = []
        for word in questions[q]['expected_response']:
            if word.lower() in content_response:
                response_assertions += 1
                palavras_encontradas.append(word)

        result.append({f"{q}": ((response_assertions/int(len(questions[q]['expected_response']))))})
        
        print(f"Palavras encontradas na resposta: {palavras_encontradas}\n-----")
        print(f"{response_assertions} de {len(questions[q]['expected_response'])} palavras esperadas encontradas na resposta.\n-----")
        print(f"{round((response_assertions/int(len(questions[q]['expected_response']))) * 100,2)}% de acerto na resposta.\n-----")

    parametro = 0.6

    validation = [r for r in result if list(r.values())[0] < parametro]
    assert len(validation) < 1, [f" Resposta para {list(q.keys())[0]} não atingiu o mínimo de {p*100}% de acerto. Aderência: {round(list(q.values())[0]*100,2)}%" for q, p in zip(validation, [parametro]*len(validation))]

