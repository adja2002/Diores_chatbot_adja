"""
Script de test pour le chatbot DIORES
Permet de tester le système sans Twilio
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(__file__))

from chatbot.memory import ConversationMemory
from chatbot.agent import DioresAgent


async def test_chatbot():
    """Test basique du chatbot"""
    print("=" * 60)
    print("Test du Chatbot DIORES")
    print("=" * 60)
    
    # Initialiser la mémoire
    print("\n1. Initialisation de la mémoire...")
    memory = ConversationMemory()
    print("[OK] Memoire initialisee")
    
    # Initialiser l'agent
    print("\n2. Initialisation de l'agent...")
    try:
        agent = DioresAgent(memory=memory)
        print("[OK] Agent initialise")
    except Exception as e:
        print(f"[ERREUR] Erreur lors de l'initialisation: {e}")
        print("\nNote: Assurez-vous d'avoir configuré MISTRAL_API_KEY dans .env")
        return
    
    # Test de conversation
    print("\n3. Test de conversation...")
    user_id = "test_user_001"
    
    messages = [
        "Bonjour",
        "J'ai fait la série S2",
        "J'ai 19 ans",
        "Mes notes: Math 14, SCPH 15, FR 12, AN 10, PHILO 11, SVT 13, HG 14",
        "je veux avoir des informations sur les formations en fst "
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"Utilisateur: {message}")
        
        try:
            response = await agent.process_message(
                user_message=message,
                user_phone=user_id
            )
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"[ERREUR] Erreur: {e}")
            import traceback
            traceback.print_exc()
    
    # Afficher le profil final
    print("\n4. Profil final:")
    profile = memory.get_profile(user_id)
    print(f"Profil: {profile}")
    
    print("\n" + "=" * 60)
    print("Test terminé")
    print("=" * 60)


if __name__ == "__main__":
    # Vérifier les variables d'environnement
    if not os.getenv("MISTRAL_API_KEY"):
        print("[ATTENTION] MISTRAL_API_KEY non definie")
        print("Créez un fichier .env avec votre clé API Mistral")
        print("\nExemple:")
        print("MISTRAL_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Lancer le test
    asyncio.run(test_chatbot())

