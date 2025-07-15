import os
import getpass
from dotenv import load_dotenv

def set_env_var(var_name: str, prompt: str) -> None:
    """
    Prompt for and set an environment variable if it is not already
    defined. Uses getpass.getpass so values won't show on-screen
    """
    if not os.environ.get(var_name):
        os.environ[var_name] = getpass.getpass(prompt)

def configure_env() -> None:
    """
    Load .env and ensure all required vars are set.
    Call this at the very top of  your app. 
    """
    # Load defaults from .env file
    load_dotenv()

    # Prompt for any missing values
    set_env_var("OPENAI_API_KEY", "Enter your OpenAI API key: ")
    for key in ["OPENAI_API_KEY"]:
        status = "Check" if os.getenv(key) else "Error"
        print(f"{key}: {status}")

if __name__ == "__main__":
    configure_env()

