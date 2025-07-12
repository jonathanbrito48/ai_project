import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

class LLMInterface:
    def __init__(self, model_name: str = "gemma:2b", temperature: float = 0.1):
        
        self.llm = ChatOllama(model=model_name, temperature=temperature, base_url="http://ollama:11434")

    def generate_response(self, messages: list) -> str:

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Erro ao gerar resposta do LLM: {e}")
            return "Não foi possível gerar uma resposta no momento."
        
