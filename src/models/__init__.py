"""
Models Package
Contains different recommendation algorithms
"""

from .collaborative_filtering import CollaborativeFilteringModel
from .content_based import ContentBasedModel

__all__ = ['CollaborativeFilteringModel', 'ContentBasedModel']