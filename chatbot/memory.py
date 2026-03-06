"""
Système de mémoire conversationnelle pour le chatbot
Gère l'historique des conversations et les profils utilisateurs
"""

import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import logging

try:
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError:
    # Fallback si langchain n'est pas installé
    HumanMessage = None
    AIMessage = None

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Gère la mémoire conversationnelle et les profils utilisateurs
    
    Stocke:
    - L'historique des messages par utilisateur
    - Le profil de chaque utilisateur (notes, série, âge, etc.)
    """
    
    def __init__(self, storage_path: str = "chatbot/data/memory"):
        """
        Initialise la mémoire conversationnelle
        
        Args:
            storage_path: Chemin pour stocker les données (optionnel, peut utiliser Redis/DB)
        """
        self.storage_path = storage_path
        self.conversations: Dict[str, List[Dict]] = {}
        self.profiles: Dict[str, Dict] = {}
        
        # Créer le dossier de stockage si nécessaire
        os.makedirs(storage_path, exist_ok=True)
        
        # Charger les données existantes
        self._load_data()
    
    def _load_data(self):
        """Charge les données depuis le stockage (si fichier)"""
        try:
            profiles_file = os.path.join(self.storage_path, "profiles.json")
            conversations_file = os.path.join(self.storage_path, "conversations.json")
            
            if os.path.exists(profiles_file):
                with open(profiles_file, "r", encoding="utf-8") as f:
                    self.profiles = json.load(f)
            
            if os.path.exists(conversations_file):
                with open(conversations_file, "r", encoding="utf-8") as f:
                    self.conversations = json.load(f)
        except Exception as e:
            logger.warning(f"Impossible de charger les données: {str(e)}")
    
    def _save_data(self):
        """Sauvegarde les données dans le stockage"""
        try:
            profiles_file = os.path.join(self.storage_path, "profiles.json")
            conversations_file = os.path.join(self.storage_path, "conversations.json")
            
            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f, ensure_ascii=False, indent=2)
            
            with open(conversations_file, "w", encoding="utf-8") as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder les données: {str(e)}")
    
    def add_message(
        self,
        user_id: str,
        role: str,
        content: str
    ):
        """
        Ajoute un message à l'historique de conversation
        
        Args:
            user_id: Identifiant unique de l'utilisateur (numéro de téléphone)
            role: "user" ou "assistant"
            content: Contenu du message
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversations[user_id].append(message)
        
        # Limiter l'historique à 50 messages par utilisateur
        if len(self.conversations[user_id]) > 50:
            self.conversations[user_id] = self.conversations[user_id][-50:]
        
        # Sauvegarder périodiquement
        if len(self.conversations[user_id]) % 10 == 0:
            self._save_data()
    
    def get_history(
        self,
        user_id: str,
        max_messages: int = 20
    ) -> List[Dict]:
        """
        Récupère l'historique de conversation d'un utilisateur
        
        Args:
            user_id: Identifiant unique de l'utilisateur
            max_messages: Nombre maximum de messages à retourner
        
        Returns:
            Liste des messages (les plus récents en dernier)
        """
        if user_id not in self.conversations:
            return []
        
        messages = self.conversations[user_id][-max_messages:]
        
        # Convertir au format LangChain
        history = []
        if HumanMessage and AIMessage:
            for msg in messages:
                if msg["role"] == "user":
                    history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    history.append(AIMessage(content=msg["content"]))
        else:
            # Fallback: retourner les messages bruts
            history = messages
        
        return history
    
    def get_conversation(self, user_id: str) -> List[Dict]:
        """
        Récupère la conversation brute d'un utilisateur (sans conversion LangChain)
        
        Args:
            user_id: Identifiant unique de l'utilisateur
        
        Returns:
            Liste des messages avec role et content
        """
        return self.conversations.get(user_id, [])
    
    def get_profile(self, user_id: str) -> Dict:
        """
        Récupère le profil d'un utilisateur
        
        Args:
            user_id: Identifiant unique de l'utilisateur
        
        Returns:
            Dictionnaire avec le profil (peut être vide)
        """
        return self.profiles.get(user_id, {})
    
    def update_profile(
        self,
        user_id: str,
        updates: Dict
    ):
        """
        Met à jour le profil d'un utilisateur
        
        Args:
            user_id: Identifiant unique de l'utilisateur
            updates: Dictionnaire avec les champs à mettre à jour
        """
        if user_id not in self.profiles:
            self.profiles[user_id] = {}
        
        # Mise à jour récursive pour les dictionnaires imbriqués (comme "notes")
        for key, value in updates.items():
            if key == "notes" and isinstance(value, dict):
                if "notes" not in self.profiles[user_id]:
                    self.profiles[user_id]["notes"] = {}
                self.profiles[user_id]["notes"].update(value)
            else:
                self.profiles[user_id][key] = value
        
        # Sauvegarder
        self._save_data()
    
    def clear_conversation(self, user_id: str):
        """Efface l'historique de conversation d'un utilisateur"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            self._save_data()
    
    def clear_profile(self, user_id: str):
        """Efface le profil d'un utilisateur"""
        if user_id in self.profiles:
            del self.profiles[user_id]
            self._save_data()

