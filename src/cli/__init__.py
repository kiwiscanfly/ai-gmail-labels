"""CLI module for the email categorization agent."""

from .base import BaseCLICommand, BaseEmailProcessor
from .main import main

__all__ = ["BaseCLICommand", "BaseEmailProcessor", "main"]