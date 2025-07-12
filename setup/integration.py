import kagglehub
import os
from .models import veiculos, SessionLocal
from sqlalchemy.exc import SQLAlchemyError
import time


class integration_engine:

    def __init__(self, kaggle_dataset: str = "farhanhossein/used-vehicles-for-sale"):
        self.dataset = kaggle_dataset
        self.session = SessionLocal()
        
    def get_dataset(self):

        get_dataset = kagglehub.dataset_download(self.dataset)
        path_dataset = str(get_dataset)

        return path_dataset
    
    def read_dataset(self, path_dataset_kaggle):

        files = [file for file in os.listdir(path_dataset_kaggle) if file.endswith('.csv')]
        df = []
        with open(f'{path_dataset_kaggle}/{files[0]}','r') as file:
            for linhas in file:
                df.append(linhas.strip().split(','))

        return df
    
    def ETL_integration_data(self, df):

        carros = []

        for carro in df[1:]:
            carros.append(
                {
                    'ano_modelo': int(carro[1]),
                    'marca': carro[2],
                    'modelo': carro[3],
                    'tipo': carro[5],
                    'tipo_motor': carro[6],
                    'transmissao': carro[7],
                    'numero_portas': carro[12],
                    'combustivel': carro[13],
                    'preco_tabela_fipe': float(carro[-1]) * 5.42 # Conversão para Real
                }
            )
        
        modelos_adicionados = set()

        for carro in carros:
            if not carro['modelo'] in modelos_adicionados:
                novo_carro = veiculos(
                    ano_modelo=carro['ano_modelo'],
                    marca=carro['marca'],
                    modelo=carro['modelo'],
                    tipo=carro['tipo'],
                    tipo_motor=carro['tipo_motor'],
                    transmissao=carro['transmissao'],
                    numero_portas=carro['numero_portas'],
                    combustivel=carro['combustivel'],
                    preco_tabela_fipe=carro['preco_tabela_fipe']
                )
                modelos_adicionados.add(carro['modelo'])
                self.session.add(novo_carro)
                os.system('clear')
                print(f"{len(modelos_adicionados)} veículos inseridos!")
                time.sleep(0)
            self.session.commit()


def integration_run():
    engine = integration_engine()
    path = engine.get_dataset()
    data = engine.read_dataset(path)
    engine.ETL_integration_data(data)

