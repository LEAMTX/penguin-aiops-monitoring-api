from pydantic import BaseModel

#les données que l’utilisateur envoie dans swagger lorsqu'il clique sur execute
class PenguinInputByUser(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float