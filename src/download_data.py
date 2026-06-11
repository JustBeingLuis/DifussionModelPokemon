import os
import requests
from tqdm import tqdm
from config import DATA_RAW_DIR

def download_pokemon_sprites(num_pokemon=1025): 

    if not os.path.exists(DATA_RAW_DIR):
        os.makedirs(DATA_RAW_DIR)
        
    print(f"Iniciando descarga de {num_pokemon} sprites en {DATA_RAW_DIR}...")
    
    base_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{}.png"
    
    for i in tqdm(range(1, num_pokemon + 1), desc="Descargando Pokémon"):
        url = base_url.format(i)
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(DATA_RAW_DIR, f"{i}.png")
            with open(file_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"\n[Aviso] No se encontró sprite para el ID {i}")

if __name__ == "__main__":
    download_pokemon_sprites(1025)
