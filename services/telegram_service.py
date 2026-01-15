"""
TelegramService: Handle Telegram webhook interactions.
"""

from typing import Dict, Any, Optional, Tuple
import aiohttp


class TelegramService:
    """
    Service for interacting with Telegram Bot API.
    """
    
    BASE_URL = "https://api.telegram.org/bot{token}"
    
    def __init__(self, bot_token: str):
        """
        Initialize TelegramService.
        
        Args:
            bot_token: Telegram Bot API token.
        """
        self.bot_token = bot_token
        self.api_url = self.BASE_URL.format(token=bot_token)
    
    def parse_update(self, update: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Parse incoming Telegram update.
        
        Args:
            update: Raw update dict from webhook.
            
        Returns:
            Tuple of (file_id, chat_id, message_text).
        """
        message = update.get("message", {})
        chat_id = message.get("chat", {}).get("id")
        text = message.get("text")
        
        # Check for photo
        photos = message.get("photo", [])
        file_id = None
        if photos:
            # Get largest photo (last in array)
            file_id = photos[-1].get("file_id")
        
        return file_id, chat_id, text
    
    async def get_file_bytes(self, file_id: str) -> bytes:
        """
        Download file from Telegram servers.
        
        Args:
            file_id: Telegram file ID.
            
        Returns:
            File bytes.
        """
        async with aiohttp.ClientSession() as session:
            # Get file path
            async with session.get(f"{self.api_url}/getFile", params={"file_id": file_id}) as resp:
                data = await resp.json()
                file_path = data["result"]["file_path"]
            
            # Download file
            file_url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
            async with session.get(file_url) as resp:
                return await resp.read()
    
    async def send_message(self, chat_id: int, text: str) -> Dict[str, Any]:
        """
        Send message to Telegram chat.
        
        Args:
            chat_id: Chat ID to send to.
            text: Message text.
            
        Returns:
            API response.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/sendMessage",
                json={"chat_id": chat_id, "text": text}
            ) as resp:
                return await resp.json()
