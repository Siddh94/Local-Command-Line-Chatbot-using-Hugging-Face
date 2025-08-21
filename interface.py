"""
Main interface module for the chatbot.
Implements the CLI loop that integrates the model and memory components.
"""

import sys
import time
from model_loader import ModelLoader
from chat_memory import ChatMemory


class ChatbotInterface:
    """Main chatbot interface that handles user interaction and coordinates components."""
    
    def __init__(self, model_name="distilgpt2", memory_turns=5):
        """
        Initialize the chatbot interface.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
            memory_turns (int): Number of conversation turns to remember
        """
        self.model_loader = ModelLoader(model_name)
        self.chat_memory = ChatMemory(memory_turns)
        self.is_running = False
        
    def display_welcome(self):
        """Display welcome message and basic information."""
        print("=" * 60)
        print("🤖 Welcome to the Local Hugging Face Chatbot!")
        print("=" * 60)
        print(f"Model: {self.model_loader.model_name}")
        print(f"Memory: Last {self.chat_memory.max_turns} conversation turns")
        print("Commands: /exit, /help, /memory, /clear")
        print("=" * 60)
        print()
    
    def display_help(self):
        """Display help information."""
        print("\n📚 Available Commands:")
        print("  /exit   - Exit the chatbot")
        print("  /help   - Show this help message")
        print("  /memory - Show conversation memory")
        print("  /clear  - Clear conversation memory")
        print("  /stats  - Show memory statistics")
        print()
    
    def display_memory(self):
        """Display the current conversation memory."""
        print("\n🧠 Conversation Memory:")
        print("-" * 40)
        print(self.chat_memory.format_for_display())
        print("-" * 40)
        
        stats = self.chat_memory.get_memory_stats()
        print(f"Memory usage: {stats['memory_usage']} turns")
        print()
    
    def display_stats(self):
        """Display memory and model statistics."""
        print("\n📊 Statistics:")
        print("-" * 30)
        
        # Memory stats
        mem_stats = self.chat_memory.get_memory_stats()
        print(f"Memory: {mem_stats['memory_usage']} turns")
        
        # Model stats
        model_info = self.model_loader.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Device: {model_info['device']}")
        print(f"Status: {'Loaded' if model_info['is_loaded'] else 'Not loaded'}")
        print("-" * 30)
        print()
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.chat_memory.clear_memory()
        print("🧹 Conversation memory cleared!")
        print()
    
    def process_command(self, user_input: str) -> bool:
        """
        Process special commands.
        
        Args:
            user_input (str): User input to check for commands
            
        Returns:
            bool: True if command was processed, False if it's a regular message
        """
        command = user_input.strip().lower()
        
        if command == "/exit":
            print("👋 Exiting chatbot. Goodbye!")
            return True
            
        elif command == "/help":
            self.display_help()
            return True
            
        elif command == "/memory":
            self.display_memory()
            return True
            
        elif command == "/clear":
            self.clear_memory()
            return True
            
        elif command == "/stats":
            self.display_stats()
            return True
            
        return False
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate a bot response using the model and memory context.
        
        Args:
            user_input (str): User's input message
            
        Returns:
            str: Bot's generated response
        """
        # Get conversation context
        context = self.chat_memory.get_recent_context(turns=3)
        
        # Create prompt with context
        if context:
            prompt = f"{context}\nUser: {user_input}\nBot:"
        else:
            prompt = f"User: {user_input}\nBot:"
        
        # Generate response
        response = self.model_loader.generate_response(prompt, max_new_tokens=40)
        
        # Clean up response
        if response.startswith("Bot:"):
            response = response[4:].strip()
        
        return response
    
    def run(self):
        """Main chatbot loop."""
        print("🔄 Loading model...")
        
        # Load the model
        if not self.model_loader.load_model():
            print("❌ Failed to load model. Exiting.")
            return
        
        print("✅ Model loaded successfully!")
        print()
        
        self.display_welcome()
        self.is_running = True
        
        while self.is_running:
            try:
                # Get user input
                user_input = input("👤 You: ").strip()
                
                # Check for empty input
                if not user_input:
                    continue
                
                # Process commands
                if self.process_command(user_input):
                    if user_input == "/exit":
                        break
                    continue
                
                # Generate response
                print("🤖 Bot: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)
                
                # Add to memory
                self.chat_memory.add_exchange(user_input, response)
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user.")
                break
            except EOFError:
                print("\n\n⚠️  End of input reached.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or use /exit to quit.")
                print()
        
        print("👋 Chatbot session ended.")


def main():
    """Main entry point for the chatbot."""
    try:
        # Create and run the chatbot with GPT-2 model
        chatbot = ChatbotInterface(model_name="gpt2", memory_turns=5)
        chatbot.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
