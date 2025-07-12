from sqlalchemy import create_engine, MetaData, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from contextlib import contextmanager


engine = create_engine('sqlite:///vehicles.db', echo=False)
metadata = MetaData()
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()


class veiculos(Base):
    __tablename__ = 'veiculos'
    id = Column(Integer, primary_key=True,autoincrement="auto")
    ano_modelo = Column(Integer, nullable=False)
    marca = Column(String, nullable=False)
    modelo = Column(String,nullable=False)
    tipo = Column(String)
    tipo_motor = Column(String, nullable=False)
    transmissao = Column(String, nullable=False)
    numero_portas = Column(String, nullable=True)
    combustivel = Column(String,nullable=False)
    preco = Column(String, nullable=False)

    def __repr__(self):
        return f"<Carro(marca='{self.modelo}')>"
    
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)


@contextmanager
def get_session():
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()


