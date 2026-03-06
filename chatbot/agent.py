"""
Agent LangChain pour le chatbot DIORES
Orchestre les interactions, la mémoire conversationnelle, le RAG et les prédictions
"""

import os
from typing import Dict, Optional, List
import logging
import asyncio

# Imports LangChain agents (peuvent échouer avec certaines versions)
try:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain.tools import Tool
except ImportError as e:
    logging.warning(f"Imports LangChain agents non disponibles: {e}")
    AgentExecutor = None
    create_react_agent = None
    Tool = None

# Imports LangChain core (toujours nécessaires)
try:
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    logging.error(f"Imports LangChain core non disponibles: {e}")
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    HumanMessage = None
    AIMessage = None

# Import Mistral AI (toujours nécessaire)
try:
    from langchain_mistralai import ChatMistralAI
except ImportError as e:
    logging.error(f"Import ChatMistralAI non disponible: {e}")
    ChatMistralAI = None

from chatbot.memory import ConversationMemory
from chatbot.rag import RAGSystem
from chatbot.diores_api import DioresAPIClient
from chatbot.profile_extractor import ProfileExtractor

logger = logging.getLogger(__name__) # pour logger les erreurs et les messages


class DioresAgent:
    """
    Agent LangChain principal qui orchestre:
    - La mémoire conversationnelle
    - L'extraction du profil utilisateur
    - La recherche sémantique (RAG) dans ChromaDB
    - Les appels aux API DIORES (classifier + lasso)
    - La génération de réponses avec Mistral
    """
    
    def __init__(self, memory: ConversationMemory):
        """
        Initialise l'agent avec tous ses composants
        
        Args:
            memory: Instance de ConversationMemory pour gérer les conversations
        """
        self.memory = memory
        self.llm = None
        self.rag_system = None
        self.diores_api = None
        self.profile_extractor = None
        
        self._initialize_components()
        self._create_agent()
    
    def _initialize_components(self):
        """Initialise tous les composants de l'agent"""
        logger.info("Initialisation des composants de l'agent...")
        
        # Initialiser le LLM Mistral
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError("MISTRAL_API_KEY non définie dans les variables d'environnement")
        
        self.llm = ChatMistralAI(
            model="mistral-small-latest",  # ou "mistral-tiny" pour plus rapide
            temperature=0.7, # pour la créativité de la réponse
            api_key=mistral_api_key # pour l'authentification avec Mistral AI
        )
        
        # Initialiser le système RAG (ChromaDB + vectorisation)
        self.rag_system = RAGSystem()
        
        # Initialiser le client API DIORES
        self.diores_api = DioresAPIClient()
        
        # Initialiser l'extracteur de profil
        self.profile_extractor = ProfileExtractor()
        
        logger.info("Composants initialisés avec succès")
    
    def _create_agent(self):
        """Crée l'agent LangChain avec ses outils"""
        if not AgentExecutor or not create_react_agent:
            logger.warning("LangChain non disponible, utilisation d'un agent simplifié")
            self.agent_executor = None
            return
        
        # Définir les outils disponibles pour l'agent
        tools = [
            Tool(
                name="get_formation_info",
                func=self._get_formation_info,
                description="Recherche des informations sur une formation via recherche sémantique dans la base de connaissances"
            ),
            Tool(
                name="get_prediction",
                func=self._get_prediction,
                description="Obtient les prédictions DIORES (probabilité d'orientation et de réussite) pour un étudiant"
            ),
            Tool(
                name="extract_profile",
                func=self._extract_profile,
                description="Extrait ou met à jour le profil de l'étudiant depuis la conversation"
            )
        ]
        
        # Template de prompt pour l'agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()), # pour le système de l'agent (le système de l'agent est stocké dans la variable "system")
            MessagesPlaceholder(variable_name="chat_history"), # pour l'historique de la conversation (l'historique de la conversation est stocké dans la variable "chat_history")
            ("human", "{input}"), # pour le message de l'utilisateur (le message de l'utilisateur est stocké dans la variable "input")
            MessagesPlaceholder(variable_name="agent_scratchpad") # pour le scratchpad de l'agent (pour les actions intermédiaires)
        ])
        
        # Créer l'agent ReAct
        agent = create_react_agent(
            llm=self.llm, # pour le LLM (le LLM est stocké dans la variable "llm")
            tools=tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True, # pour la verbosité (True pour afficher les messages intermédiaires, False pour ne pas afficher les messages intermédiaires)
            max_iterations=5, # pour le nombre maximum d'itérations (5 pour 5 itérations)
            handle_parsing_errors=True # pour gérer les erreurs de parsing (True pour gérer les erreurs de parsing, False pour ne pas gérer les erreurs de parsing)
        )
    
    def _get_system_prompt(self) -> str:
        """Retourne le prompt système pour l'agent"""
        return """Tu es un assistant d'orientation intelligent pour les bacheliers sénégalais.
                Tu travailles avec le système DIORES de l'Université Cheikh Anta Diop (UCAD).

                TON RÔLE:
                - Aider les étudiants à comprendre les formations disponibles à la Faculté des Sciences et Techniques (FST)
                - Collecter leurs informations (série BAC, notes, âge, etc.)
                - Leur fournir des prédictions personnalisées sur leurs chances d'orientation et de réussite
                - Répondre de manière naturelle, empathique et encourageante

                FORMATIONS DISPONIBLES:
                - L1MPI (Mathématiques, Physique, Informatique)
                - L1BCGS (Biologie, Chimie, Géologie, Sciences)
                - L1PCSM (Physique, Chimie, Sciences Mathématiques)

                INFORMATIONS À COLLECTER:
                - Série du BAC (S1, S2, ou S3)
                - Notes du BAC (MATH, SCPH, FR, AN, PHILO, SVT, HG, EPS)
                - Âge
                - Sexe
                - Résidence
                - Académie d'origine

                QUAND TU AS LE PROFIL COMPLET:
                1. Utilise get_prediction pour obtenir les prédictions DIORES
                2. Utilise get_formation_info pour obtenir des informations sur la formation
                3. Génère une réponse personnalisée, claire et encourageante

                IMPORTANT:
                - Sois patient et pose les questions une par une
                - Valide les informations reçues
                - Explique les résultats de manière compréhensible
                - Encourage l'étudiant même si les prédictions sont modérées"""
    
    async def process_message(
        self,
        user_message: str,
        user_phone: str
    ) -> str:
        """
        Traite un message utilisateur et génère une réponse
        
        Args:
            user_message: Message de l'utilisateur
            user_phone: Numéro de téléphone (identifiant unique)
        
        Returns:
            Réponse générée par l'agent
        """
        try:
            # Récupérer l'historique de conversation
            chat_history = self.memory.get_history(user_phone)
            
            # Obtenir le dernier message de l'assistant pour le contexte
            last_assistant_message = None
            conversation_history = self.memory.get_conversation(user_phone)
            if conversation_history and len(conversation_history) > 0:
                # Chercher le dernier message de l'assistant
                for msg in reversed(conversation_history):
                    if msg.get("role") == "assistant":
                        last_assistant_message = msg.get("content", "")
                        break
            
            # Vérifier si le profil est complet
            profile = self.memory.get_profile(user_phone)
            is_profile_complete = self._check_profile_completeness(profile)
            
            # Toujours essayer d'extraire des informations du message (même si le profil semble complet)
            # pour s'assurer qu'on a les dernières informations
            extracted_info = self.profile_extractor.extract(
                user_message, 
                profile,
                context=last_assistant_message
            )
            if extracted_info:
                self.memory.update_profile(user_phone, extracted_info)
                profile = self.memory.get_profile(user_phone)
                # Re-vérifier après mise à jour
                is_profile_complete = self._check_profile_completeness(profile)
            
            # Log pour déboguer
            logger.info(f"Profil après extraction - complet: {is_profile_complete}, formation: {profile.get('formation')}, notes: {list(profile.get('notes', {}).keys()) if profile.get('notes') else 'None'}")
            
            # Préparer le contexte pour l'agent
            context = self._prepare_context(profile, is_profile_complete)
            
            # Construire le message avec contexte
            full_message = f"{context}\n\nMessage de l'utilisateur: {user_message}"
            
            # Exécuter l'agent
            if self.agent_executor:
                response = await self.agent_executor.ainvoke({
                    "input": full_message,
                    "chat_history": chat_history
                })
                response_text = response.get("output", "Désolé, je n'ai pas pu générer de réponse.")
            else:
                # Fallback: utiliser directement le LLM
                # Passer le message original, pas le message avec contexte
                logger.info("Utilisation de _simple_llm_response (fallback)")
                try:
                    response_text = await self._simple_llm_response(user_message, profile, is_profile_complete)
                    logger.info(f"Réponse générée par _simple_llm_response (longueur: {len(response_text)})")
                except Exception as e:
                    logger.error(f"Erreur dans _simple_llm_response: {e}", exc_info=True)
                    if is_profile_complete:
                     profil_msg = "Votre profil semble complet."
                    else:
                      profil_msg = "Pour vous aider, j'ai besoin de plus d'informations sur votre profil."
                    response_text = ("Bonjour ! Je suis votre assistant d'orientation DIORES. "
                                      "J'ai bien reçu votre message. "
                                    f"{profil_msg} "
                                    "Pouvez-vous reformuler votre question ?"
                                    )

            
            # Sauvegarder dans la mémoire
            if response_text:
                self.memory.add_message(user_phone, "user", user_message)
                self.memory.add_message(user_phone, "assistant", response_text)
            else:
                logger.error("ATTENTION: response_text est vide ou None!")
                response_text = "Désolé, je n'ai pas pu générer de réponse. Veuillez réessayer."
            
            logger.info(f"Réponse finale retournée (longueur: {len(response_text)})")
            return response_text
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {str(e)}", exc_info=True)
            import traceback
            logger.error(f"Traceback complet: {traceback.format_exc()}")
            return f"Désolé, une erreur s'est produite lors du traitement de votre message. Veuillez réessayer. Erreur: {str(e)}"
    
    def _check_profile_completeness(self, profile: Dict) -> bool:
        """Vérifie si le profil utilisateur est complet pour les prédictions DIORES"""
        # Champs strictement nécessaires pour les prédictions DIORES
        required_fields = [
            "serie", "notes", "age", "sexe", "formation"
        ]
        
        if not profile:
            return False
        
        # Vérifier les champs obligatoires
        for field in required_fields:
            if field not in profile or profile[field] is None:
                return False
        
        # Vérifier que les notes essentielles sont présentes
        # Les notes minimales requises pour les modèles DIORES
        if "notes" in profile:
            required_notes = ["MATH", "SCPH", "FR", "AN", "PHILO"]
            notes = profile.get("notes", {})
            for note in required_notes:
                if note not in notes or notes[note] is None:
                    return False
        
        return True
    
    def _prepare_context(self, profile: Dict, is_complete: bool) -> str:
        """Prépare le contexte pour l'agent"""
        if not profile:
            return "L'utilisateur n'a pas encore fourni d'informations. Commence par le saluer et lui demander sa série BAC."
        
        context_parts = []
        
        if is_complete:
            context_parts.append("PROFIL UTILISATEUR COMPLET:")
            context_parts.append(f"- Série: {profile.get('serie')}")
            context_parts.append(f"- Âge: {profile.get('age')}")
            context_parts.append(f"- Sexe: {profile.get('sexe')}")
            context_parts.append(f"- Formation souhaitée: {profile.get('formation')}")
            context_parts.append(f"- Notes: {profile.get('notes')}")
        else:
            context_parts.append("PROFIL UTILISATEUR INCOMPLET:")
            for key, value in profile.items():
                if value:
                    context_parts.append(f"- {key}: {value}")
            context_parts.append("\nTu dois collecter les informations manquantes une par une.")
        
        return "\n".join(context_parts)
    
    # Outils de l'agent
    def _get_formation_info(self, query: str) -> str:
        """Recherche des informations sur une formation via RAG"""
        try:
            # Détecter le type de requête pour optimiser la recherche
            query_type = self.rag_system.detect_query_type(query)
            results = self.rag_system.search(query, top_k=3, source=query_type if query_type != "both" else None)
            if results:
                return "\n\n".join([r["content"] for r in results])
            return "Aucune information trouvée sur cette formation."
        except Exception as e:
            logger.error(f"Erreur dans get_formation_info: {str(e)}")
            return f"Erreur lors de la recherche: {str(e)}"
    
    def _get_prediction(self, formation: str) -> str:
        """Obtient les prédictions DIORES pour un étudiant"""
        try:
            # Cette fonction sera appelée par l'agent, mais on a besoin du profil
            # On va utiliser un profil temporaire depuis la mémoire
            # En pratique, l'agent devrait passer le profil en paramètre
            return "Utilise extract_profile d'abord pour obtenir le profil complet."
        except Exception as e:
            logger.error(f"Erreur dans get_prediction: {str(e)}")
            return f"Erreur lors de la prédiction: {str(e)}"
    
    def _extract_profile(self, message: str) -> str:
        """Extrait les informations du profil depuis un message"""
        try:
            extracted = self.profile_extractor.extract(message, {})
            if extracted:
                return f"Informations extraites: {extracted}"
            return "Aucune information extraite."
        except Exception as e:
            logger.error(f"Erreur dans extract_profile: {str(e)}")
            return f"Erreur lors de l'extraction: {str(e)}"
    
    async def get_prediction_for_profile(
        self,
        profile: Dict,
        formation: str
    ) -> Dict:
        """
        Obtient les prédictions DIORES pour un profil complet
        
        Args:
            profile: Profil complet de l'étudiant
            formation: Formation souhaitée (L1MPI, L1BCGS, L1PCSM)
        
        Returns:
            Dictionnaire avec les prédictions
        """
        try:
            # Appeler l'API DIORES
            predictions = await self.diores_api.predict(
                profile=profile,
                formation=formation
            )
            
            return predictions
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise
    
    async def _simple_llm_response(
        self,
        message: str,
        profile: Dict,
        is_complete: bool
    ) -> str:
        """
        Réponse simplifiée sans agent (fallback)
        
        Args:
            message: Message de l'utilisateur
            profile: Profil utilisateur
            is_complete: Si le profil est complet
        
        Returns:
            Réponse générée
        """
        try:
            message_lower = message.lower()
            
            # Détecter le type de question
            formation_keywords = ["mpi", "bcgs", "pcsm", "licence", "formation", "information", "débouché", "programme"]
            university_keywords = ["université", "universités", "thies", "thiès", "ucad", "ugb", "uidt", "uadb", "uasz", "ussein", "uam", "vie étudiante"]
            
            # Mots-clés pour les prédictions personnalisées (doivent être combinés avec des pronoms personnels)
            prediction_personal_keywords = ["mes chances", "mon profil", "ma chance", "mes résultats", "peux-je", "puis-je", "ai-je", "suis-je"]
            prediction_general_keywords = ["chances", "probabilité", "prédiction", "prédire"]
            
            is_formation_query = any(keyword in message_lower for keyword in formation_keywords)
            is_university_query = any(keyword in message_lower for keyword in university_keywords)
            
            # Détecter si c'est une demande de prédiction PERSONNELLE (nécessite profil complet)
            # vs une question d'information générale sur l'admission
            is_prediction_query = False
            if any(keyword in message_lower for keyword in prediction_personal_keywords):
                is_prediction_query = True
                logger.info("Demande de prédiction détectée via prediction_personal_keywords")
            elif any(keyword in message_lower for keyword in prediction_general_keywords):
                # Vérifier si c'est une question personnelle avec des pronoms
                personal_pronouns = ["mes", "mon", "ma", "je", "moi", "mon profil", "mes notes"]
                if any(pronoun in message_lower for pronoun in personal_pronouns):
                    is_prediction_query = True
                    logger.info("Demande de prédiction détectée via prediction_general_keywords + pronoms personnels")
                # Sinon, c'est probablement une question générale sur les conditions d'admission
            
            # PRIORITÉ 1: Détecter si c'est une demande de prédiction AVANT de traiter les questions générales
            # Détecter si c'est une demande de prédiction (même si is_prediction_query n'est pas détecté, vérifier "chance" ou "admission")
            is_prediction_request = (
                is_prediction_query or 
                "chance" in message_lower or 
                "prédiction" in message_lower or
                ("admission" in message_lower and ("mes" in message_lower or "mon" in message_lower or "je" in message_lower))
            )
            
            logger.info(f"is_prediction_query: {is_prediction_query}, is_prediction_request: {is_prediction_request}, is_complete: {is_complete}, formation: {profile.get('formation') if profile else 'None'}")
            logger.info(f"Profil détaillé: {profile}")
            
            # PRIORITÉ ABSOLUE: Si le profil est complet ET que c'est une demande de prédiction, faire les prédictions immédiatement
            # Vérifier aussi si le message contient toutes les infos nécessaires même si le profil n'est pas encore sauvegardé
            has_all_info = (
                profile.get("serie") and 
                profile.get("formation") and 
                profile.get("age") and 
                profile.get("sexe") and 
                profile.get("notes") and
                all(note in profile.get("notes", {}) for note in ["MATH", "SCPH", "FR", "AN", "PHILO"])
            )
            
            if (is_complete or has_all_info) and profile.get("formation") and is_prediction_request:
                logger.info(f"=== DÉBUT CALCUL PRÉDICTIONS ===")
                logger.info(f"Profil complet détecté, calcul des prédictions DIORES pour {profile.get('formation')}")
                logger.info(f"Détails du profil: série={profile.get('serie')}, notes={list(profile.get('notes', {}).keys())}")
                
                try:
                    # Calculer les prédictions
                    logger.info("Appel à get_prediction_for_profile...")
                    predictions = await self.get_prediction_for_profile(
                        profile, profile["formation"]
                    )
                    logger.info(f"Prédictions calculées avec succès: {predictions}")
                    
                    if not predictions:
                        raise ValueError("Aucune prédiction retournée")
                    
                    # Extraire les valeurs des prédictions
                    prob_orientation = predictions.get('prob_orientation', 0)
                    prob_reussite = predictions.get('prob_reussite', 0)
                    
                    # Construire la réponse directement avec les valeurs numériques (sans attendre le LLM)
                    logger.info("Construction de la réponse avec les prédictions...")
                    
                    # Formater la mention de manière plus informative
                    mention_display = predictions.get('mention', 'N/A')
                    if mention_display is None or mention_display == 'None' or mention_display == '':
                        if predictions.get('session') == 'Deuxième Session':
                            mention_display = "Non applicable (Deuxième Session)"
                        else:
                            mention_display = "À déterminer"
                    
                    # Formater la probabilité d'orientation avec une note si nécessaire
                    orientation_display = f"{prob_orientation:.1f}%"
                    if prob_orientation >= 99.0:
                        orientation_display += " (très élevée)"
                    elif prob_orientation >= 80.0:
                        orientation_display += " (élevée)"
                    elif prob_orientation >= 60.0:
                        orientation_display += " (bonne)"
                    elif prob_orientation >= 40.0:
                        orientation_display += " (modérée)"
                    else:
                        orientation_display += " (faible)"
                    
                    formation_name = profile.get('formation', 'N/A')
                    newline = '\n'
                    
                    predictions_header = f"""Voici vos prédictions DIORES pour la formation {formation_name} :

- Probabilité d'orientation : {orientation_display}
- Probabilité de réussite en L1 : {prob_reussite:.1f}%
- Admission prédite : {predictions.get('admission', 'N/A')}
- Session prédite : {predictions.get('session', 'N/A')}
- Mention prédite : {mention_display}

"""
                    
                    # Rechercher des infos sur la formation (rapide, sans bloquer)
                    formation_info_text = ""
                    try:
                        formation_info = self.rag_system.search(profile["formation"], top_k=2, source="fst")
                        if formation_info:
                            formation_info_text = "\n".join([r["content"] for r in formation_info[:2]])
                    except Exception as e:
                        logger.warning(f"Erreur RAG (non bloquant): {e}")
                    
                    # TOUJOURS utiliser la réponse directe avec les valeurs numériques exactes
                    # Ne pas faire confiance au LLM pour afficher les bonnes valeurs
                    logger.info("Utilisation de la réponse directe avec les valeurs numériques exactes")
                    
                    model_used = predictions.get('model_used', 'Doc1')
                    serie = profile.get('serie', 'N/A')
                    age = profile.get('age', 'N/A')
                    sexe = profile.get('sexe', 'N/A')
                    notes = profile.get('notes', {})
                    math_note = notes.get('MATH', 'N/A')
                    scph_note = notes.get('SCPH', 'N/A')
                    fr_note = notes.get('FR', 'N/A')
                    an_note = notes.get('AN', 'N/A')
                    philo_note = notes.get('PHILO', 'N/A')
                    
                    formation_info_section = ""
                    if formation_info_text:
                        formation_info_section = f"{newline}Informations sur la formation :{newline}{formation_info_text}"
                    
                    response_text = predictions_header + f"""Ces prédictions sont basées sur votre profil académique et les modèles DIORES V4 (Modèle utilisé: {model_used}).

Votre profil :
- Série : {serie}
- Âge : {age} ans
- Sexe : {sexe}
- Notes principales : 
  • Mathématiques : {math_note}/20
  • Sciences Physiques : {scph_note}/20
  • Français : {fr_note}/20
  • Anglais : {an_note}/20
  • Philosophie : {philo_note}/20
{formation_info_section}

Ces résultats sont des estimations basées sur des données historiques et des modèles d'apprentissage automatique. Continuez à travailler pour maximiser vos chances de réussite !
"""
                    
                    logger.info(f"=== FIN CALCUL PRÉDICTIONS - Réponse générée (longueur: {len(response_text)}) ===")
                    return response_text
                    
                except Exception as e:
                    logger.error(f"ERREUR CRITIQUE lors de la prédiction: {str(e)}", exc_info=True)
                    import traceback
                    logger.error(f"Traceback complet: {traceback.format_exc()}")
                    # Réponse d'erreur détaillée pour déboguer
                    notes_keys = list(profile.get('notes', {}).keys()) if profile.get('notes') else 'Aucune'
                    return f"""Désolé, une erreur s'est produite lors du calcul de tes prédictions.

Erreur : {str(e)}

Ton profil semble complet :
- Série : {profile.get('serie', 'N/A')}
- Formation : {profile.get('formation', 'N/A')}
- Notes : {notes_keys}

Peux-tu réessayer ou me donner plus de détails sur ton profil ?"""
            
            # PRIORITÉ 2: Si c'est une demande de prédiction mais le profil n'est pas complet, demander les infos manquantes
            if is_prediction_query and not is_complete:
                missing = self._get_missing_info(profile)
                if missing:
                    missing_str = ', '.join(missing)
                    return f"""Pour te donner une réponse précise sur tes chances personnalisées avec le système DIORES, j'ai besoin de compléter ton profil.

Informations manquantes : {missing_str}

Peux-tu me fournir ces informations ? Une fois que j'aurai toutes tes notes du BAC, je pourrai utiliser le système DIORES pour te donner des prédictions précises sur tes chances d'admission et de réussite dans la formation que tu souhaites."""
            
            # PRIORITÉ 3: Si le profil n'est pas complet ET que ce n'est pas une question d'information générale,
            # demander les informations manquantes une par une
            if not is_complete and not is_formation_query and not is_university_query and not is_prediction_query:
                missing = self._get_missing_info(profile)
                if missing:
                    # Demander une information à la fois pour être plus clair
                    next_info = missing[0]
                    return f"Pour continuer, j'ai besoin de connaître : {next_info}. Peux-tu me le fournir ?"
            
            # Réponse générique pour les salutations et questions générales
            logger.info("Génération d'une réponse générique avec le LLM")
            profile_status = "OUI" if is_complete else "NON"
            profile_str = str(profile) if profile else "Aucun profil encore"
            
            prompt = f"""Tu es un assistant d'orientation pour les bacheliers sénégalais. 
            Tu travailles avec le système DIORES de l'Université Cheikh Anta Diop (UCAD).

            TON RÔLE PRINCIPAL :
            - Répondre aux questions d'information générale sur les formations FST (MPI, BCGS, PCSM) et les universités
            - Collecter les informations du profil UNIQUEMENT si l'utilisateur demande des prédictions personnalisées
            - Utiliser le système DIORES pour faire des prédictions précises sur les chances d'admission et de réussite

            Message utilisateur: {message}
            Profil actuel: {profile_str}
            Profil complet: {profile_status}

            INSTRUCTIONS IMPORTANTES :
            - Si le profil est COMPLET et que l'utilisateur demande ses chances, tu DOIS utiliser le système DIORES pour faire les prédictions (ne demande PAS d'informations supplémentaires)
            - Si c'est une salutation, salue chaleureusement et explique que tu peux aider avec l'orientation
            - Si c'est une question d'information générale, réponds directement sans demander le profil complet
            - Si l'utilisateur demande ses chances personnalisées ("mes chances", "peux-je", etc.) mais le profil est incomplet, demande UNIQUEMENT les informations manquantes essentielles (série, âge, sexe, formation, notes en Mathématiques, Sciences Physiques, Français, Anglais, Philosophie)
            - Ne demande JAMAIS d'informations supplémentaires comme "lycée d'origine", "rang de classement", ou "choix de filière secondaire" - ces informations ne sont pas nécessaires pour les prédictions DIORES
            - Réponds de manière naturelle, encourageante et en français
            - sois toujour respectueux et utilise l'humour pour répondre aux questions 
            - Ne répond pas sous forme de markdown et n'utilise pas des  caractéres bizzares
            - N'utilise jamais ces caractères #,*  dans  les réponses et  dans les questions  """
                        
            response = await self.llm.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            logger.error(f"Erreur dans _simple_llm_response: {str(e)}", exc_info=True)
            return "Désolé, une erreur s'est produite. Pouvez-vous reformuler votre question ?"
    
    def _get_missing_info(self, profile: Dict) -> List[str]:
        """Retourne la liste des informations manquantes pour les prédictions DIORES"""
        missing = []
        required = {
            "serie": "votre série BAC (S1, S2 ou S3)",
            "age": "votre âge",
            "sexe": "votre sexe",
            "formation": "la formation souhaitée (MPI, BCGS ou PCSM)"
        }
        
        for key, label in required.items():
            if key not in profile or not profile[key]:
                missing.append(label)
        
        # Vérifier les notes essentielles pour les prédictions DIORES
        notes = profile.get("notes", {})
        required_notes = {
            "MATH": "Mathématiques",
            "SCPH": "Sciences Physiques",
            "FR": "Français",
            "AN": "Anglais",
            "PHILO": "Philosophie"
        }
        
        for note_key, note_label in required_notes.items():
            if note_key not in notes or notes[note_key] is None:
                missing.append(f"votre note en {note_label}")
        
        return missing
    
    def _build_response_prompt(
        self,
        profile: Dict,
        predictions: Dict,
        formation_info: str,
        user_message: str
    ) -> str:
        """Construit le prompt pour générer une réponse personnalisée"""
        prob_orientation = predictions.get("prob_orientation", 0)
        prob_reussite = predictions.get("prob_reussite", 0)
        admission = predictions.get("admission", "N/A")
        session = predictions.get("session", "N/A")
        mention = predictions.get("mention", "N/A")
        model_used = predictions.get("model_used", "Doc1")
        
        # Formater les notes de manière lisible
        notes_str = ", ".join([f"{matiere}: {note}" for matiere, note in profile.get('notes', {}).items()])
        
        formation_name = profile.get('formation', 'N/A')
        
        prompt = f"""Tu es un assistant d'orientation intelligent pour les bacheliers sénégalais utilisant le système DIORES de l'UCAD.

            PROFIL DE L'ÉTUDIANT:
            - Série BAC: {profile.get('serie', 'N/A')}
            - Âge: {profile.get('age', 'N/A')} ans
            - Formation souhaitée: {formation_name}
            - Notes du BAC: {notes_str if notes_str else 'Non spécifiées'}

            PRÉDICTIONS DIORES (calculées par les modèles DIORES V4 - Modèle utilisé: {model_used}):
            - Probabilité d'orientation (chances d'être admis dans la formation): {prob_orientation:.1f}%
            - Probabilité de réussite en L1 (chances de valider la première année): {prob_reussite:.1f}%
            - Admission prédite: {admission}
            - Session prédite: {session}
            - Mention prédite: {mention}

            INFORMATIONS SUR LA FORMATION:
            {formation_info}

            MESSAGE DE L'ÉTUDIANT: {user_message}

            INSTRUCTIONS STRICTES POUR TA RÉPONSE:
            
            OBLIGATOIRE - Tu DOIS commencer ta réponse par afficher EXACTEMENT ces informations dans ce format :
            
            "Voici vos prédictions DIORES pour la formation {formation_name} :
            
            • Probabilité d'orientation : {prob_orientation:.1f}%
            • Probabilité de réussite en L1 : {prob_reussite:.1f}%
            • Admission prédite : {admission}
            • Session prédite : {session}
            • Mention prédite : {mention}"
            
            Ensuite, tu peux :
            1. Expliquer ce que signifient ces probabilités (orientation = chances d'être admis, réussite = chances de valider la L1)
            2. Interpréter les résultats de manière encourageante et honnête
            3. Donner des conseils pratiques basés sur ces résultats précis
            4. Utiliser les informations sur la formation pour donner du contexte
            
            RÈGLES STRICTES :
            - N'utilise JAMAIS de termes vagues comme "très bonnes", "élevées", "modérées" sans donner les valeurs numériques
            - Affiche TOUJOURS les valeurs exactes des probabilités ({prob_orientation:.1f}% et {prob_reussite:.1f}%)
            - Les prédictions DIORES sont précises et doivent être communiquées telles quelles
            - Sois naturel, empathique, professionnel et utilise un ton encourageant"""
                
        return prompt