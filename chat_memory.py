"""
Chat memory module for the chatbot.
Implements a sliding window memory buffer to maintain conversation context.
"""

from collections import deque
from typing import List, Tuple, Optional


class ChatMemory:
    """Implements a sliding window memory buffer for conversation history."""
    
    def __init__(self, max_turns: int = 5):
        """
        Initialize the chat memory.
        
        Args:
            max_turns (int): Maximum number of conversation turns to remember
        """
        self.max_turns = max_turns
        self.conversation_history = deque(maxlen=max_turns * 2)  # *2 because each turn has user + bot
        
    def add_exchange(self, user_input: str, bot_response: str):
        """
        Add a user-bot exchange to the memory.
        
        Args:
            user_input (str): User's input message
            bot_response (str): Bot's response message
        """
        self.conversation_history.append(("user", user_input))
        self.conversation_history.append(("bot", bot_response))
    
    def get_context(self) -> str:
        """
        Get the conversation context as a formatted string for the model.
        
        Returns:
            str: Formatted conversation context
        """
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for speaker, message in self.conversation_history:
            if speaker == "user":
                context_parts.append(f"User: {message}")
            else:
                context_parts.append(f"Bot: {message}")
        
        return "\n".join(context_parts)
    
    def get_recent_context(self, turns: int = 3) -> str:
        """
        Get the most recent conversation context.
        
        Args:
            turns (int): Number of recent turns to include
            
        Returns:
            str: Formatted recent conversation context
        """
        if not self.conversation_history:
            return ""
        
        # Get the last N turns (N user inputs + N bot responses)
        recent_exchanges = list(self.conversation_history)[-turns * 2:]
        
        context_parts = []
        for speaker, message in recent_exchanges:
            if speaker == "user":
                context_parts.append(f"User: {message}")
            else:
                context_parts.append(f"Bot: {message}")
        
        return "\n".join(context_parts)
    
    def get_full_history(self) -> List[Tuple[str, str]]:
        """
        Get the full conversation history.
        
        Returns:
            List[Tuple[str, str]]: List of (speaker, message) tuples
        """
        return list(self.conversation_history)
    
    def clear_memory(self):
        """Clear all conversation history."""
        self.conversation_history.clear()
    
    def get_memory_stats(self) -> dict:
        """
        Get statistics about the current memory usage.
        
        Returns:
            dict: Memory statistics
        """
        total_exchanges = len(self.conversation_history) // 2
        return {
            "max_turns": self.max_turns,
            "current_turns": total_exchanges,
            "memory_usage": f"{total_exchanges}/{self.max_turns}",
            "is_full": total_exchanges >= self.max_turns
        }
    
    def format_for_display(self) -> str:
        """
        Format the conversation history for display purposes.
        
        Returns:
            str: Formatted conversation history
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        formatted_lines = []
        for i, (speaker, message) in enumerate(self.conversation_history):
            if speaker == "user":
                formatted_lines.append(f"👤 User: {message}")
            else:
                formatted_lines.append(f"🤖 Bot: {message}")
        
        return "\n".join(formatted_lines)
