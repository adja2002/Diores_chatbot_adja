
"""
Chargeur simple pour les chunks de formations depuis un fichier JSON
Remplace corpus_loader.py pour simplifier le code
"""

import json
import os
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class JSONLoader:
    """
    Chargeur simple pour charger les chunks depuis un fichier JSON
    
    Structure attendue du JSON:
    [
        {
            "id": "chunk_id",
            "text": "contenu du chunk",
            "metadata": {
                "formation": "MPI|BCGS|PCSM|general",
                "section": "nom_section",
                "universite": "nom_universite" (optionnel)
            }
        },
        ...
    ]
    """
    
    def __init__(self, json_path: Optional[str] = None):
        """
        Initialise le chargeur JSON
        
        Args:
            json_path: Chemin vers le fichier JSON. Si None, utilise le chemin par défaut
        """
        if json_path is None:
            # Chemin par défaut : chatbot/corpus/formations_chunks.json
            base_dir = os.path.dirname(os.path.dirname(__file__))
            json_path = os.path.join(base_dir, "chatbot", "corpus", "formations_chunks.json")
        
        self.json_path = json_path
        self._validate_path()
    
    def _validate_path(self):
        """Vérifie que le fichier JSON existe"""
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(
                f"Fichier JSON introuvable : {self.json_path}\n"
                f"Assurez-vous que le fichier formations_chunks.json existe dans chatbot/corpus/"
            )
    
    def load_chunks(self) -> List[Dict]:
        """
        Charge tous les chunks depuis le fichier JSON
        
        Returns:
            Liste de dictionnaires avec les chunks chargés
        """
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            
            logger.info(f"Chargement de {len(chunks)} chunks depuis {self.json_path}")
            
            # Valider la structure
            validated_chunks = []
            for idx, chunk in enumerate(chunks):
                if self._validate_chunk(chunk, idx):
                    validated_chunks.append(chunk)
            
            logger.info(f"{len(validated_chunks)} chunks validés sur {len(chunks)}")
            return validated_chunks
            
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de parsing JSON : {str(e)}")
            raise ValueError(f"Le fichier JSON n'est pas valide : {str(e)}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du JSON : {str(e)}")
            raise
    
    def _validate_chunk(self, chunk: Dict, index: int) -> bool:
        """
        Valide qu'un chunk a la structure attendue
        
        Args:
            chunk: Chunk à valider
            index: Index du chunk dans la liste (pour les messages d'erreur)
        
        Returns:
            True si le chunk est valide, False sinon
        """
        required_fields = ['id', 'text', 'metadata']
        
        for field in required_fields:
            if field not in chunk:
                logger.warning(f"Chunk {index} invalide : champ '{field}' manquant. Ignoré.")
                return False
        
        # Vérifier que metadata contient au moins 'formation'
        if 'formation' not in chunk['metadata']:
            logger.warning(f"Chunk {index} ({chunk.get('id', 'unknown')}) invalide : "
                          f"metadata.formation manquant. Ignoré.")
            return False
        
        return True
    
    def get_chunks_by_formation(self, formation: str) -> List[Dict]:
        """
        Récupère tous les chunks d'une formation spécifique
        
        Args:
            formation: Nom de la formation (MPI, BCGS, PCSM, general)
        
        Returns:
            Liste des chunks de cette formation
        """
        all_chunks = self.load_chunks()
        return [
            chunk for chunk in all_chunks
            if chunk.get('metadata', {}).get('formation', '').upper() == formation.upper()
        ]
    
    def get_fst_chunks(self) -> List[Dict]:
        """
        Récupère tous les chunks des formations FST (MPI, BCGS, PCSM)
        Exclut les chunks 'general' qui ne sont pas spécifiques à une formation
        
        Returns:
            Liste des chunks des formations FST
        """
        all_chunks = self.load_chunks()
        fst_formations = ['MPI', 'BCGS', 'PCSM']
        return [
            chunk for chunk in all_chunks
            if chunk.get('metadata', {}).get('formation', '').upper() in fst_formations
        ]
    
    def get_general_chunks(self) -> List[Dict]:
        """
        Récupère tous les chunks généraux (formation='general')
        Ces chunks contiennent des informations sur les universités, la vie étudiante, etc.
        
        Returns:
            Liste des chunks généraux
        """
        all_chunks = self.load_chunks()
        return [
            chunk for chunk in all_chunks
            if chunk.get('metadata', {}).get('formation', '').lower() == 'general'
        ]
    
    def get_statistics(self) -> Dict:
        """
        Retourne des statistiques sur les chunks chargés
        
        Returns:
            Dictionnaire avec les statistiques
        """
        chunks = self.load_chunks()
        
        stats = {
            'total_chunks': len(chunks),
            'by_formation': {},
            'by_section': {},
            'with_universite': 0
        }
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            # Compter par formation
            formation = metadata.get('formation', 'unknown')
            stats['by_formation'][formation] = stats['by_formation'].get(formation, 0) + 1
            
            # Compter par section
            section = metadata.get('section', 'unknown')
            stats['by_section'][section] = stats['by_section'].get(section, 0) + 1
            
            # Compter ceux avec universite
            if 'universite' in metadata:
                stats['with_universite'] += 1
        
        return stats

