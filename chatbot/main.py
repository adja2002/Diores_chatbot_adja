"""
Serveur FastAPI principal pour le chatbot DIORES WhatsApp
Gère les webhooks Twilio et orchestre les interactions avec l'agent LangChain
"""

import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports
# Cela permet de lancer depuis le répertoire chatbot/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import logging

from chatbot.agent import DioresAgent
from chatbot.memory import ConversationMemory

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# Initialiser FastAPI
app = FastAPI(
    title="DIORES Chatbot API",
    description="API pour le chatbot d'orientation des bacheliers via WhatsApp",
    version="1.0.0"
)

# Initialiser les composants globaux
agent = None
memory = None

@app.on_event("startup")
async def startup_event():
    """Initialise les composants au démarrage du serveur"""
    global agent, memory
    
    logger.info("Initialisation du chatbot DIORES...")
    
    try:
        # Initialiser la mémoire conversationnelle
        memory = ConversationMemory()
        
        # Initialiser l'agent LangChain
        agent = DioresAgent(memory=memory)
        
        logger.info("Chatbot DIORES initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {str(e)}")
        raise


@app.get("/")
async def root():
    """Endpoint de santé"""
    return {
        "status": "ok",
        "service": "DIORES Chatbot API",
        "version": "1.0.0"
    }


@app.post("/")
async def root_post(request: Request):
    """
    Endpoint POST pour compatibilité avec certaines configurations Twilio
    Redirige vers le webhook WhatsApp
    """
    return await whatsapp_webhook(request)


@app.get("/health")
async def health_check():
    """Vérification de l'état du service"""
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "memory_initialized": memory is not None
    }


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request):
    """
    Webhook pour recevoir les messages WhatsApp via Twilio
    
    Flow:
    1. Twilio envoie le message via webhook
    2. On extrait le numéro et le message
    3. On transmet à l'agent LangChain
    4. L'agent génère une réponse
    5. On renvoie la réponse à Twilio
    """
    try:
        # Récupérer les données du formulaire Twilio
        form_data = await request.form()
        incoming_message = form_data.get("Body", "").strip()
        from_number = form_data.get("From", "").strip()
        
        # Nettoyer le numéro (enlever le préfixe whatsapp:)
        if from_number.startswith("whatsapp:"):
            from_number = from_number.replace("whatsapp:", "")
        
        logger.info(f"Message reçu de {from_number}: {incoming_message}")
        
        # Vérifier si c'est une commande de réinitialisation
        if incoming_message and incoming_message.upper() in ["RESET", "RECOMMENCER", "RECOMMENCE", "NOUVEAU", "NOUVELLE CONVERSATION"]:
            if memory is not None:
                memory.clear_conversation(from_number)
                memory.clear_profile(from_number)
                logger.info(f"Conversation réinitialisée pour {from_number}")
                
                twilio_response = MessagingResponse()
                twilio_response.message(
                    "Conversation reinitialisee ! Je suis pret pour une nouvelle discussion. "
                    "Bonjour ! Je suis votre assistant d'orientation DIORES. Comment puis-je vous aider ?"
                )
                return Response(
                    content=str(twilio_response),
                    media_type="application/xml"
                )
        
        if not incoming_message:
            response_text = "Bonjour ! Je suis votre assistant d'orientation DIORES. Comment puis-je vous aider ?"
        else:
            # Traiter le message avec l'agent LangChain
            if agent is None:
                logger.error("Agent non initialisé!")
                response_text = "Désolé, le système est temporairement indisponible. Veuillez réessayer dans quelques instants."
            else:
                try:
                    logger.info(f"Traitement du message avec l'agent pour {from_number}")
                    # Obtenir la réponse de l'agent avec timeout implicite
                    response_text = await agent.process_message(
                        user_message=incoming_message,
                        user_phone=from_number
                    )
                    logger.info(f"Réponse obtenue de l'agent (longueur: {len(response_text) if response_text else 0})")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement par l'agent: {str(e)}", exc_info=True)
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    response_text = f"Désolé, une erreur s'est produite lors du traitement de votre message. Veuillez réessayer. Erreur: {str(e)}"
        
        # S'assurer qu'on a toujours une réponse
        if not response_text or len(response_text.strip()) == 0:
            logger.error("ATTENTION: response_text est vide après traitement!")
            response_text = "Bonjour ! Je suis votre assistant d'orientation DIORES. J'ai bien reçu votre message. Pouvez-vous reformuler votre question ?"
        
        # Créer la réponse Twilio
        twilio_response = MessagingResponse()
        twilio_response.message(response_text)
        
        logger.info(f"Réponse envoyée à {from_number} (longueur: {len(response_text)})")
        
        return Response(
            content=str(twilio_response),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Erreur dans le webhook WhatsApp: {str(e)}", exc_info=True)
        
        # Réponse d'erreur pour l'utilisateur
        twilio_response = MessagingResponse()
        twilio_response.message(
            "Désolé, une erreur s'est produite. Veuillez réessayer plus tard."
        )
        
        return Response(
            content=str(twilio_response),
            media_type="application/xml",
            status_code=500
        )


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Endpoint API alternatif pour tester le chatbot sans Twilio
    """
    try:
        data = await request.json()
        user_message = data.get("message", "")
        user_id = data.get("user_id", "test_user")
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message requis")
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent non initialisé")
        
        response_text = await agent.process_message(
            user_message=user_message,
            user_phone=user_id
        )
        
        return {
            "response": response_text,
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset_conversation(request: Request):
    """
    Réinitialise la conversation et le profil d'un utilisateur
    Utile pour recommencer une discussion depuis le début
    """
    try:
        data = await request.json()
        user_id = data.get("user_id") or data.get("user_phone")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id ou user_phone requis")
        
        if memory is None:
            raise HTTPException(status_code=503, detail="Mémoire non initialisée")
        
        # Effacer la conversation et le profil
        memory.clear_conversation(user_id)
        memory.clear_profile(user_id)
        
        logger.info(f"Conversation et profil réinitialisés pour {user_id}")
        
        return {
            "status": "success",
            "message": f"Conversation et profil réinitialisés pour {user_id}",
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

