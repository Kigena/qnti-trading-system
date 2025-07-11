#!/usr/bin/env python3
"""
QNTI LLM+MCP Integration Setup Script
Helps install dependencies and configure the LLM integration
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QNTILLMSetup:
    """Setup helper for QNTI LLM+MCP integration"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.requirements_file = self.project_root / "requirements_llm.txt"
        self.config_file = self.project_root / "qnti_llm_config.json"
        self.memory_dir = self.project_root / "qnti_memory"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            return False
        logger.info(f"Python version: {sys.version}")
        return True
    
    def install_dependencies(self):
        """Install required dependencies"""
        logger.info("Installing LLM+MCP dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], check=True)
            logger.info("Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def check_ollama_installation(self):
        """Check if Ollama is installed and running"""
        logger.info("Checking Ollama installation...")
        
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Ollama is installed and accessible")
                logger.info("Available models:")
                print(result.stdout)
                return True
            else:
                logger.warning("Ollama command failed")
                return False
        except FileNotFoundError:
            logger.warning("Ollama not found in PATH")
            return False
    
    def install_ollama_model(self, model_name="llama3"):
        """Install the specified Ollama model"""
        logger.info(f"Installing Ollama model: {model_name}")
        
        try:
            subprocess.run(["ollama", "pull", model_name], check=True)
            logger.info(f"Model {model_name} installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install model {model_name}: {e}")
            return False
        except FileNotFoundError:
            logger.error("Ollama not found. Please install Ollama first.")
            return False
    
    def create_memory_directory(self):
        """Create the ChromaDB memory directory"""
        logger.info("Creating memory directory...")
        
        try:
            self.memory_dir.mkdir(exist_ok=True)
            logger.info(f"Memory directory created: {self.memory_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create memory directory: {e}")
            return False
    
    def setup_configuration(self):
        """Setup configuration files"""
        logger.info("Setting up configuration...")
        
        # Check if config exists
        if self.config_file.exists():
            logger.info("Configuration file already exists")
            return True
        
        # Create default config
        default_config = {
            "llm": {
                "model": "llama3",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 30
            },
            "chroma": {
                "path": "./qnti_memory",
                "collection_name": "qnti_context",
                "persist_directory": "./qnti_memory"
            },
            "news": {
                "api_key": "",
                "update_interval": 60,
                "sources": ["reuters", "bloomberg", "cnbc"],
                "queries": ["forex trading", "market analysis", "economic indicators"]
            },
            "market_data": {
                "symbols": ["SPY", "QQQ", "DXY", "GLD", "BTC-USD", "EURUSD", "GBPUSD", "USDJPY"],
                "update_interval": 30
            },
            "scheduling": {
                "daily_brief_hour": 6,
                "daily_brief_minute": 0,
                "news_update_interval": 60,
                "market_data_interval": 30,
                "context_cleanup_hour": 2
            },
            "security": {
                "api_key": "qnti-secret-key",
                "jwt_secret": "qnti-jwt-secret-2024",
                "token_expiry_hours": 24
            },
            "integration": {
                "trade_log_path": "./trade_log.csv",
                "open_trades_path": "./open_trades.json",
                "backup_path": "./qnti_backups",
                "enable_auto_context": True,
                "context_window_size": 20
            },
            "features": {
                "enable_news_analysis": True,
                "enable_market_sentiment": True,
                "enable_trade_correlation": True,
                "enable_performance_insights": True,
                "enable_risk_assessment": True
            },
            "limits": {
                "max_context_documents": 1000,
                "max_daily_requests": 10000,
                "max_response_length": 5000,
                "context_retention_days": 30
            }
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Configuration file created: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to create configuration: {e}")
            return False
    
    def test_integration(self):
        """Test the LLM integration"""
        logger.info("Testing LLM integration...")
        
        try:
            # Test imports
            import chromadb
            import ollama
            logger.info("✓ ChromaDB and Ollama imports successful")
            
            # Test ChromaDB
            client = chromadb.PersistentClient(path=str(self.memory_dir))
            collection = client.get_or_create_collection("test_collection")
            logger.info("✓ ChromaDB connection successful")
            
            # Test Ollama
            ollama_client = ollama.Client(host="http://localhost:11434")
            models = ollama_client.list()
            logger.info(f"✓ Ollama connection successful, {len(models.get('models', []))} models available")
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
    
    def run_setup(self):
        """Run the complete setup process"""
        logger.info("Starting QNTI LLM+MCP Integration Setup")
        logger.info("=" * 50)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Installing dependencies", self.install_dependencies),
            ("Creating memory directory", self.create_memory_directory),
            ("Setting up configuration", self.setup_configuration),
            ("Checking Ollama installation", self.check_ollama_installation),
            ("Testing integration", self.test_integration)
        ]
        
        failed_steps = []
        
        for step_name, step_func in steps:
            logger.info(f"\n{step_name}...")
            if not step_func():
                failed_steps.append(step_name)
                logger.error(f"✗ {step_name} failed")
            else:
                logger.info(f"✓ {step_name} completed")
        
        logger.info("\n" + "=" * 50)
        
        if failed_steps:
            logger.error("Setup completed with errors:")
            for step in failed_steps:
                logger.error(f"  - {step}")
            
            # Provide manual instructions
            logger.info("\nManual setup instructions:")
            if "Checking Ollama installation" in failed_steps:
                logger.info("1. Install Ollama from: https://ollama.ai/download")
                logger.info("2. Run: ollama pull llama3")
            
            if "Installing dependencies" in failed_steps:
                logger.info("3. Install dependencies manually:")
                logger.info(f"   pip install -r {self.requirements_file}")
            
            return False
        else:
            logger.info("✓ Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Configure your News API key in qnti_llm_config.json")
            logger.info("2. Start your QNTI system: python qnti_main_system.py")
            logger.info("3. Test LLM endpoints at: http://localhost:5000/llm/status")
            return True

def main():
    """Main setup function"""
    setup = QNTILLMSetup()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "install-model":
            model = sys.argv[2] if len(sys.argv) > 2 else "llama3"
            setup.install_ollama_model(model)
        elif command == "test":
            setup.test_integration()
        elif command == "deps":
            setup.install_dependencies()
        else:
            print("Usage:")
            print("  python setup_llm_integration.py          # Run full setup")
            print("  python setup_llm_integration.py deps     # Install dependencies only")
            print("  python setup_llm_integration.py test     # Test integration")
            print("  python setup_llm_integration.py install-model [model_name]  # Install Ollama model")
    else:
        # Run full setup
        success = setup.run_setup()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 