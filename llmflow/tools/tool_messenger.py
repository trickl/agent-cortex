"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Messenger Tool - Enables messaging capabilities across different platforms with support for various message formats and delivery protocols.
"""

import os
import json
import time
import logging
import hashlib
import tempfile
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import requests
from urllib.parse import urlencode
import base64
import mimetypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLMFlow registration decorator
from llmflow.tools.tool_decorator import register_tool

# Optional imports with fallbacks
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    from slack_sdk.web import SlackResponse
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

try:
    import discord
    from discord.ext import commands
    DISCORD_PY_AVAILABLE = True
except ImportError:
    DISCORD_PY_AVAILABLE = False

try:
    import telegram
    from telegram import Bot, Update, InlineKeyboardMarkup, InlineKeyboardButton
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

@dataclass
class PlatformConfig:
    
    platform: str
    token: str
    webhook_url: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0

@dataclass
class SlackConfig:
    """Slack-specific configuration."""
    token: str = os.getenv('SLACK_BOT_TOKEN', '')
    platform: str = 'slack'
    webhook_url: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    signing_secret: Optional[str] = os.getenv('SLACK_SIGNING_SECRET', None)
    app_token: Optional[str] = os.getenv('SLACK_APP_TOKEN', None)
    oauth_token: Optional[str] = os.getenv('SLACK_OAUTH_TOKEN', None)

@dataclass
class DiscordConfig:
    """Discord-specific configuration."""
    token: str = os.getenv('DISCORD_BOT_TOKEN', '')
    platform: str = 'discord'
    webhook_url: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    guild_id: Optional[str] = os.getenv('DISCORD_GUILD_ID', None)
    intents: List[str] = None
    command_prefix: str = '!'

@dataclass
class TelegramConfig:
    """Telegram-specific configuration."""
    token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    platform: str = 'telegram'
    webhook_url: Optional[str] = None
    api_base_url: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    rate_limit_delay: float = 1.0
    webhook_secret: Optional[str] = os.getenv('TELEGRAM_WEBHOOK_SECRET', None)
    chat_id: Optional[str] = os.getenv('TELEGRAM_CHAT_ID', None)
    parse_mode: str = 'HTML'

@dataclass
class Message:
    """Universal message format."""
    id: str
    platform: str
    channel_id: str
    user_id: str
    username: str
    content: str
    timestamp: str
    message_type: str
    attachments: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    thread_id: Optional[str] = None
    reply_to: Optional[str] = None

@dataclass
class Channel:
    """Universal channel/chat format."""
    id: str
    name: str
    platform: str
    type: str  # public, private, dm, group
    member_count: int
    created_at: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = None

class SlackIntegration:
    """
    Slack integration with comprehensive API support.
    """
    
    def __init__(self, config: SlackConfig):
        self.config = config
        self.client = None
        self.rtm_client = None
        self.message_history = []
        self.event_handlers = {}
        
        if SLACK_SDK_AVAILABLE and config.token:
            self.client = WebClient(token=config.token)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Slack API connection."""
        try:
            if not self.client:
                return {'status': 'error', 'message': 'Slack SDK not available or token not provided'}
            
            response = self.client.auth_test()
            
            return {
                'status': 'success',
                'user_id': response['user_id'],
                'user': response['user'],
                'team': response['team'],
                'team_id': response['team_id'],
                'url': response['url']
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Connection failed: {str(e)}'}
    
    def send_message(self, channel: str, text: str, **kwargs) -> Dict[str, Any]:
        """
        Send message to Slack channel.
        
        Args:
            channel: Channel ID or name
            text: Message text
            **kwargs: Additional parameters (blocks, attachments, thread_ts, etc.)
        """
        try:
            if not self.client:
                return {'status': 'error', 'message': 'Slack client not initialized'}
            
            # Prepare message parameters
            params = {
                'channel': channel,
                'text': text,
                **kwargs
            }
            
            response = self.client.chat_postMessage(**params)
            
            # Record in history
            self.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'channel': channel,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'message_ts': response['ts'],
                'success': True
            })
            
            return {
                'status': 'success',
                'message_ts': response['ts'],
                'channel': response['channel'],
                'text': text,
                'permalink': self._get_permalink(channel, response['ts'])
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send message: {str(e)}'}
    
    def send_rich_message(self, channel: str, blocks: List[Dict[str, Any]] = None,
                         attachments: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Send rich message with blocks and attachments."""
        try:
            params = {
                'channel': channel,
                **kwargs
            }
            
            if blocks:
                params['blocks'] = blocks
            if attachments:
                params['attachments'] = attachments
            
            response = self.client.chat_postMessage(**params)
            
            return {
                'status': 'success',
                'message_ts': response['ts'],
                'channel': response['channel']
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send rich message: {str(e)}'}
    
    def upload_file(self, channel: str, file_path: str, title: str = None,
                   comment: str = None) -> Dict[str, Any]:
        """Upload file to Slack channel."""
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            filename = os.path.basename(file_path)
            
            response = self.client.files_upload_v2(
                channel=channel,
                file=file_path,
                title=title or filename,
                initial_comment=comment
            )
            
            return {
                'status': 'success',
                'file_id': response['file']['id'],
                'file_url': response['file']['url_private'],
                'filename': filename,
                'size': response['file']['size']
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to upload file: {str(e)}'}
    
    def get_channels(self, types: str = 'public_channel,private_channel') -> Dict[str, Any]:
        """Get list of channels."""
        try:
            response = self.client.conversations_list(types=types, limit=1000)
            
            channels = []
            for channel in response['channels']:
                channels.append(Channel(
                    id=channel['id'],
                    name=channel['name'],
                    platform='slack',
                    type='private' if channel['is_private'] else 'public',
                    member_count=channel.get('num_members', 0),
                    created_at=str(channel['created']),
                    description=channel.get('topic', {}).get('value', ''),
                    metadata={
                        'is_archived': channel.get('is_archived', False),
                        'is_general': channel.get('is_general', False)
                    }
                ))
            
            return {
                'status': 'success',
                'channels': [asdict(c) for c in channels],
                'total_channels': len(channels)
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get channels: {str(e)}'}
    
    def get_messages(self, channel: str, limit: int = 100, oldest: str = None) -> Dict[str, Any]:
        """Get messages from channel."""
        try:
            params = {'channel': channel, 'limit': limit}
            if oldest:
                params['oldest'] = oldest
            
            response = self.client.conversations_history(**params)
            
            messages = []
            for msg in response['messages']:
                messages.append(Message(
                    id=msg.get('ts', ''),
                    platform='slack',
                    channel_id=channel,
                    user_id=msg.get('user', ''),
                    username=self._get_username(msg.get('user', '')),
                    content=msg.get('text', ''),
                    timestamp=msg.get('ts', ''),
                    message_type=msg.get('type', 'message'),
                    attachments=msg.get('files', []),
                    metadata={
                        'thread_ts': msg.get('thread_ts'),
                        'reactions': msg.get('reactions', [])
                    },
                    thread_id=msg.get('thread_ts'),
                    reply_to=msg.get('reply_to')
                ))
            
            return {
                'status': 'success',
                'messages': [asdict(m) for m in messages],
                'total_messages': len(messages),
                'has_more': response.get('has_more', False)
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get messages: {str(e)}'}
    
    def create_channel(self, name: str, is_private: bool = False) -> Dict[str, Any]:
        """Create new channel."""
        try:
            response = self.client.conversations_create(
                name=name,
                is_private=is_private
            )
            
            channel = response['channel']
            
            return {
                'status': 'success',
                'channel_id': channel['id'],
                'channel_name': channel['name'],
                'is_private': channel['is_private'],
                'created': channel['created']
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create channel: {str(e)}'}
    
    def add_reaction(self, channel: str, timestamp: str, emoji: str) -> Dict[str, Any]:
        """Add reaction to message."""
        try:
            self.client.reactions_add(
                channel=channel,
                timestamp=timestamp,
                name=emoji
            )
            
            return {
                'status': 'success',
                'emoji': emoji,
                'channel': channel,
                'timestamp': timestamp
            }
            
        except SlackApiError as e:
            return {'status': 'error', 'message': f'Slack API error: {e.response["error"]}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to add reaction: {str(e)}'}
    
    def _get_username(self, user_id: str) -> str:
        """Get username by user ID."""
        try:
            response = self.client.users_info(user=user_id)
            return response['user']['name']
        except:
            return user_id
    
    def _get_permalink(self, channel: str, message_ts: str) -> str:
        """Get permalink for message."""
        try:
            response = self.client.chat_getPermalink(
                channel=channel,
                message_ts=message_ts
            )
            return response['permalink']
        except:
            return ''

class DiscordIntegration:
    """
    Discord integration with bot functionality.
    """
    
    def __init__(self, config: DiscordConfig):
        self.config = config
        self.client = None
        self.bot = None
        self.message_history = []
        self.event_handlers = {}
        self.is_running = False
        
        if DISCORD_PY_AVAILABLE and config.token:
            intents = discord.Intents.default()
            if config.intents:
                for intent in config.intents:
                    setattr(intents, intent, True)
            
            self.client = discord.Client(intents=intents)
            self.bot = commands.Bot(command_prefix=config.command_prefix, intents=intents)
    
    def send_message_sync(self, channel_id: str, content: str, **kwargs) -> Dict[str, Any]:
        """Send message synchronously using HTTP API."""
        try:
            url = f'https://discord.com/api/v10/channels/{channel_id}/messages'
            headers = {
                'Authorization': f'Bot {self.config.token}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'content': content,
                **kwargs
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=self.config.timeout)
            response.raise_for_status()
            
            message_data = response.json()
            
            # Record in history
            self.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'channel_id': channel_id,
                'content': content[:100] + '...' if len(content) > 100 else content,
                'message_id': message_data['id'],
                'success': True
            })
            
            return {
                'status': 'success',
                'message_id': message_data['id'],
                'channel_id': channel_id,
                'content': content,
                'timestamp': message_data['timestamp']
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send message: {str(e)}'}
    
    def send_embed(self, channel_id: str, title: str, description: str = None,
                   color: int = 0x00ff00, fields: List[Dict[str, Any]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Send rich embed message."""
        try:
            embed_data = {
                'title': title,
                'color': color
            }
            
            if description:
                embed_data['description'] = description
            
            if fields:
                embed_data['fields'] = fields
            
            # Add additional embed properties
            for key in ['url', 'timestamp', 'footer', 'image', 'thumbnail', 'author']:
                if key in kwargs:
                    embed_data[key] = kwargs[key]
            
            return self.send_message_sync(channel_id, '', embeds=[embed_data])
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send embed: {str(e)}'}
    
    def upload_file(self, channel_id: str, file_path: str, content: str = None) -> Dict[str, Any]:
        """Upload file to Discord channel."""
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            url = f'https://discord.com/api/v10/channels/{channel_id}/messages'
            headers = {
                'Authorization': f'Bot {self.config.token}'
            }
            
            files = {'file': open(file_path, 'rb')}
            data = {}
            
            if content:
                data['content'] = content
            
            response = requests.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            
            message_data = response.json()
            
            return {
                'status': 'success',
                'message_id': message_data['id'],
                'filename': os.path.basename(file_path),
                'attachments': message_data.get('attachments', [])
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to upload file: {str(e)}'}
        finally:
            try:
                files['file'].close()
            except:
                pass
    
    def get_guilds(self) -> Dict[str, Any]:
        """Get list of guilds (servers)."""
        try:
            url = 'https://discord.com/api/v10/users/@me/guilds'
            headers = {
                'Authorization': f'Bot {self.config.token}'
            }
            
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()
            
            guilds = response.json()
            
            return {
                'status': 'success',
                'guilds': guilds,
                'total_guilds': len(guilds)
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get guilds: {str(e)}'}
    
    def get_channels(self, guild_id: str = None) -> Dict[str, Any]:
        """Get channels for a guild."""
        try:
            target_guild = guild_id or self.config.guild_id
            if not target_guild:
                return {'status': 'error', 'message': 'Guild ID not provided'}
            
            url = f'https://discord.com/api/v10/guilds/{target_guild}/channels'
            headers = {
                'Authorization': f'Bot {self.config.token}'
            }
            
            response = requests.get(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()
            
            channels_data = response.json()
            
            channels = []
            for channel in channels_data:
                channel_type_map = {0: 'text', 2: 'voice', 4: 'category', 5: 'news'}
                
                channels.append(Channel(
                    id=channel['id'],
                    name=channel['name'],
                    platform='discord',
                    type=channel_type_map.get(channel['type'], 'unknown'),
                    member_count=0,  # Discord doesn't provide this easily
                    created_at=channel.get('created_timestamp', ''),
                    description=channel.get('topic', ''),
                    metadata={
                        'guild_id': target_guild,
                        'position': channel.get('position', 0),
                        'nsfw': channel.get('nsfw', False)
                    }
                ))
            
            return {
                'status': 'success',
                'channels': [asdict(c) for c in channels],
                'total_channels': len(channels),
                'guild_id': target_guild
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get channels: {str(e)}'}
    
    def get_messages(self, channel_id: str, limit: int = 50, before: str = None) -> Dict[str, Any]:
        """Get messages from channel."""
        try:
            url = f'https://discord.com/api/v10/channels/{channel_id}/messages'
            headers = {
                'Authorization': f'Bot {self.config.token}'
            }
            
            params = {'limit': limit}
            if before:
                params['before'] = before
            
            response = requests.get(url, headers=headers, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            messages_data = response.json()
            
            messages = []
            for msg in messages_data:
                messages.append(Message(
                    id=msg['id'],
                    platform='discord',
                    channel_id=channel_id,
                    user_id=msg['author']['id'],
                    username=msg['author']['username'],
                    content=msg['content'],
                    timestamp=msg['timestamp'],
                    message_type=msg['type'],
                    attachments=msg.get('attachments', []),
                    metadata={
                        'embeds': msg.get('embeds', []),
                        'reactions': msg.get('reactions', []),
                        'pinned': msg.get('pinned', False)
                    },
                    thread_id=msg.get('thread', {}).get('id'),
                    reply_to=msg.get('referenced_message', {}).get('id')
                ))
            
            return {
                'status': 'success',
                'messages': [asdict(m) for m in messages],
                'total_messages': len(messages)
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get messages: {str(e)}'}
    
    def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> Dict[str, Any]:
        """Add reaction to message."""
        try:
            url = f'https://discord.com/api/v10/channels/{channel_id}/messages/{message_id}/reactions/{emoji}/@me'
            headers = {
                'Authorization': f'Bot {self.config.token}'
            }
            
            response = requests.put(url, headers=headers, timeout=self.config.timeout)
            response.raise_for_status()
            
            return {
                'status': 'success',
                'emoji': emoji,
                'channel_id': channel_id,
                'message_id': message_id
            }
            
        except requests.RequestException as e:
            return {'status': 'error', 'message': f'HTTP request failed: {str(e)}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to add reaction: {str(e)}'}

class TelegramIntegration:
    """
    Telegram integration with bot functionality.
    """
    
    def __init__(self, config: TelegramConfig):
        self.config = config
        self.bot = None
        self.message_history = []
        self.event_handlers = {}
        
        if TELEGRAM_AVAILABLE and config.token:
            self.bot = Bot(token=config.token)
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Telegram bot connection."""
        try:
            if not self.bot:
                return {'status': 'error', 'message': 'Telegram bot not available or token not provided'}
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            me = loop.run_until_complete(self.bot.get_me())
            loop.close()
            
            return {
                'status': 'success',
                'bot_id': me.id,
                'username': me.username,
                'first_name': me.first_name,
                'can_join_groups': me.can_join_groups,
                'can_read_all_group_messages': me.can_read_all_group_messages
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Connection failed: {str(e)}'}
    
    def send_message(self, chat_id: str, text: str, parse_mode: str = None,
                    reply_markup: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Send message to Telegram chat."""
        try:
            if not self.bot:
                return {'status': 'error', 'message': 'Telegram bot not initialized'}
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Prepare keyboard if provided
            keyboard = None
            if reply_markup:
                if reply_markup.get('type') == 'inline':
                    buttons = []
                    for row in reply_markup.get('buttons', []):
                        button_row = []
                        for button in row:
                            button_row.append(InlineKeyboardButton(
                                text=button['text'],
                                callback_data=button.get('callback_data'),
                                url=button.get('url')
                            ))
                        buttons.append(button_row)
                    keyboard = InlineKeyboardMarkup(buttons)
            
            message = loop.run_until_complete(
                self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=parse_mode or self.config.parse_mode,
                    reply_markup=keyboard,
                    **kwargs
                )
            )
            loop.close()
            
            # Record in history
            self.message_history.append({
                'timestamp': datetime.now().isoformat(),
                'chat_id': chat_id,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'message_id': message.message_id,
                'success': True
            })
            
            return {
                'status': 'success',
                'message_id': message.message_id,
                'chat_id': chat_id,
                'text': text,
                'date': message.date.isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send message: {str(e)}'}
    
    def send_photo(self, chat_id: str, photo_path: str, caption: str = None) -> Dict[str, Any]:
        """Send photo to Telegram chat."""
        try:
            if not os.path.exists(photo_path):
                return {'status': 'error', 'message': f'Photo not found: {photo_path}'}
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            with open(photo_path, 'rb') as photo:
                message = loop.run_until_complete(
                    self.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=caption
                    )
                )
            loop.close()
            
            return {
                'status': 'success',
                'message_id': message.message_id,
                'chat_id': chat_id,
                'file_id': message.photo[-1].file_id,
                'caption': caption
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send photo: {str(e)}'}
    
    def send_document(self, chat_id: str, document_path: str, caption: str = None) -> Dict[str, Any]:
        """Send document to Telegram chat."""
        try:
            if not os.path.exists(document_path):
                return {'status': 'error', 'message': f'Document not found: {document_path}'}
            
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            with open(document_path, 'rb') as document:
                message = loop.run_until_complete(
                    self.bot.send_document(
                        chat_id=chat_id,
                        document=document,
                        caption=caption
                    )
                )
            loop.close()
            
            return {
                'status': 'success',
                'message_id': message.message_id,
                'chat_id': chat_id,
                'file_id': message.document.file_id,
                'filename': message.document.file_name,
                'file_size': message.document.file_size
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send document: {str(e)}'}
    
    def get_updates(self, offset: int = None, limit: int = 100) -> Dict[str, Any]:
        """Get updates from Telegram."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            updates = loop.run_until_complete(
                self.bot.get_updates(offset=offset, limit=limit)
            )
            loop.close()
            
            messages = []
            for update in updates:
                if update.message:
                    msg = update.message
                    messages.append(Message(
                        id=str(msg.message_id),
                        platform='telegram',
                        channel_id=str(msg.chat.id),
                        user_id=str(msg.from_user.id) if msg.from_user else '',
                        username=msg.from_user.username if msg.from_user else '',
                        content=msg.text or '',
                        timestamp=msg.date.isoformat(),
                        message_type='text',
                        attachments=[],
                        metadata={
                            'chat_type': msg.chat.type,
                            'chat_title': msg.chat.title,
                            'update_id': update.update_id
                        },
                        reply_to=str(msg.reply_to_message.message_id) if msg.reply_to_message else None
                    ))
            
            return {
                'status': 'success',
                'messages': [asdict(m) for m in messages],
                'total_updates': len(updates),
                'last_update_id': updates[-1].update_id if updates else None
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get updates: {str(e)}'}
    
    def get_chat(self, chat_id: str) -> Dict[str, Any]:
        """Get chat information."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            chat = loop.run_until_complete(
                self.bot.get_chat(chat_id=chat_id)
            )
            loop.close()
            
            chat_type_map = {
                'private': 'dm',
                'group': 'group',
                'supergroup': 'group',
                'channel': 'channel'
            }
            
            return {
                'status': 'success',
                'chat': {
                    'id': str(chat.id),
                    'type': chat_type_map.get(chat.type, chat.type),
                    'title': chat.title,
                    'username': chat.username,
                    'description': chat.description,
                    'member_count': getattr(chat, 'member_count', 0)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get chat: {str(e)}'}
    
    def set_webhook(self, webhook_url: str, secret_token: str = None) -> Dict[str, Any]:
        """Set webhook for receiving updates."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(
                self.bot.set_webhook(
                    url=webhook_url,
                    secret_token=secret_token
                )
            )
            loop.close()
            
            return {
                'status': 'success',
                'webhook_url': webhook_url,
                'result': result
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to set webhook: {str(e)}'}
    
    def delete_webhook(self) -> Dict[str, Any]:
        """Delete webhook."""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            result = loop.run_until_complete(self.bot.delete_webhook())
            loop.close()
            
            return {
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete webhook: {str(e)}'}

class MessagingPlatformsTool:
    """
    Unified tool for managing multiple messaging platforms.
    """
    
    def __init__(self):
        self.platforms = {}
        self.configs = {}
        self.operation_history = []
        self.webhook_handlers = {}
        
        # Unified statistics
        self.stats = {
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'platform_usage': {},
            'last_activity': None
        }
    
    def add_platform(self, platform_name: str, config: Union[SlackConfig, DiscordConfig, TelegramConfig]) -> Dict[str, Any]:
        """Add messaging platform configuration."""
        try:
            self.configs[platform_name] = config
            
            if config.platform == 'slack':
                self.platforms[platform_name] = SlackIntegration(config)
            elif config.platform == 'discord':
                self.platforms[platform_name] = DiscordIntegration(config)
            elif config.platform == 'telegram':
                self.platforms[platform_name] = TelegramIntegration(config)
            else:
                return {'status': 'error', 'message': f'Unsupported platform: {config.platform}'}
            
            # Initialize stats for platform
            self.stats['platform_usage'][platform_name] = {
                'messages_sent': 0,
                'messages_received': 0,
                'last_used': None,
                'errors': 0
            }
            
            return {
                'status': 'success',
                'platform': config.platform,
                'platform_name': platform_name,
                'message': f'Platform {platform_name} ({config.platform}) added successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to add platform: {str(e)}'}
    
    def send_message(self, platform_name: str, channel: str, message: str, **kwargs) -> Dict[str, Any]:
        """Send message via specified platform."""
        try:
            if platform_name not in self.platforms:
                return {'status': 'error', 'message': f'Platform {platform_name} not configured'}
            
            platform = self.platforms[platform_name]
            config = self.configs[platform_name]
            
            # Send message based on platform type
            if config.platform == 'slack':
                result = platform.send_message(channel, message, **kwargs)
            elif config.platform == 'discord':
                result = platform.send_message_sync(channel, message, **kwargs)
            elif config.platform == 'telegram':
                result = platform.send_message(channel, message, **kwargs)
            else:
                return {'status': 'error', 'message': f'Unsupported platform: {config.platform}'}
            
            # Update statistics
            if result['status'] == 'success':
                self.stats['total_messages_sent'] += 1
                self.stats['platform_usage'][platform_name]['messages_sent'] += 1
                self.stats['platform_usage'][platform_name]['last_used'] = datetime.now().isoformat()
                self.stats['last_activity'] = datetime.now().isoformat()
            else:
                self.stats['platform_usage'][platform_name]['errors'] += 1
            
            # Record operation
            self.operation_history.append({
                'type': 'send_message',
                'platform': platform_name,
                'timestamp': datetime.now().isoformat(),
                'channel': channel,
                'success': result['status'] == 'success',
                'error': result.get('message') if result['status'] == 'error' else None
            })
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to send message: {str(e)}'}
    
    def send_to_multiple_platforms(self, platforms: List[str], channel_mapping: Dict[str, str],
                                  message: str, **kwargs) -> Dict[str, Any]:
        """Send message to multiple platforms simultaneously."""
        try:
            results = {}
            successful_sends = 0
            failed_sends = 0
            
            for platform_name in platforms:
                if platform_name not in channel_mapping:
                    results[platform_name] = {
                        'status': 'error',
                        'message': f'No channel mapping for platform {platform_name}'
                    }
                    failed_sends += 1
                    continue
                
                channel = channel_mapping[platform_name]
                result = self.send_message(platform_name, channel, message, **kwargs)
                results[platform_name] = result
                
                if result['status'] == 'success':
                    successful_sends += 1
                else:
                    failed_sends += 1
            
            return {
                'status': 'success',
                'results': results,
                'summary': {
                    'total_platforms': len(platforms),
                    'successful_sends': successful_sends,
                    'failed_sends': failed_sends,
                    'success_rate': (successful_sends / len(platforms)) * 100 if platforms else 0
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Multi-platform send failed: {str(e)}'}
    
    def upload_file(self, platform_name: str, channel: str, file_path: str,
                   caption: str = None, **kwargs) -> Dict[str, Any]:
        """Upload file via specified platform."""
        try:
            if platform_name not in self.platforms:
                return {'status': 'error', 'message': f'Platform {platform_name} not configured'}
            
            platform = self.platforms[platform_name]
            config = self.configs[platform_name]
            
            # Upload file based on platform type
            if config.platform == 'slack':
                result = platform.upload_file(channel, file_path, comment=caption, **kwargs)
            elif config.platform == 'discord':
                result = platform.upload_file(channel, file_path, content=caption, **kwargs)
            elif config.platform == 'telegram':
                # Determine file type for Telegram
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    result = platform.send_photo(channel, file_path, caption=caption)
                else:
                    result = platform.send_document(channel, file_path, caption=caption)
            else:
                return {'status': 'error', 'message': f'Unsupported platform: {config.platform}'}
            
            # Record operation
            self.operation_history.append({
                'type': 'upload_file',
                'platform': platform_name,
                'timestamp': datetime.now().isoformat(),
                'channel': channel,
                'file_path': file_path,
                'success': result['status'] == 'success'
            })
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to upload file: {str(e)}'}
    
    def get_messages(self, platform_name: str, channel: str, limit: int = 50, **kwargs) -> Dict[str, Any]:
        """Get messages from specified platform and channel."""
        try:
            if platform_name not in self.platforms:
                return {'status': 'error', 'message': f'Platform {platform_name} not configured'}
            
            platform = self.platforms[platform_name]
            config = self.configs[platform_name]
            
            # Get messages based on platform type
            if config.platform == 'slack':
                result = platform.get_messages(channel, limit=limit, **kwargs)
            elif config.platform == 'discord':
                result = platform.get_messages(channel, limit=limit, **kwargs)
            elif config.platform == 'telegram':
                result = platform.get_updates(limit=limit, **kwargs)
            else:
                return {'status': 'error', 'message': f'Unsupported platform: {config.platform}'}
            
            # Update statistics
            if result['status'] == 'success':
                message_count = len(result.get('messages', []))
                self.stats['total_messages_received'] += message_count
                self.stats['platform_usage'][platform_name]['messages_received'] += message_count
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get messages: {str(e)}'}
    
    def get_channels(self, platform_name: str, **kwargs) -> Dict[str, Any]:
        """Get channels/chats for specified platform."""
        try:
            if platform_name not in self.platforms:
                return {'status': 'error', 'message': f'Platform {platform_name} not configured'}
            
            platform = self.platforms[platform_name]
            config = self.configs[platform_name]
            
            # Get channels based on platform type
            if config.platform == 'slack':
                result = platform.get_channels(**kwargs)
            elif config.platform == 'discord':
                result = platform.get_channels(**kwargs)
            elif config.platform == 'telegram':
                # Telegram doesn't have a direct equivalent, return empty list
                result = {
                    'status': 'success',
                    'channels': [],
                    'total_channels': 0,
                    'message': 'Telegram uses direct chat IDs instead of channel lists'
                }
            else:
                return {'status': 'error', 'message': f'Unsupported platform: {config.platform}'}
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get channels: {str(e)}'}
    
    def test_all_connections(self) -> Dict[str, Any]:
        """Test connections to all configured platforms."""
        try:
            results = {}
            
            for platform_name, platform in self.platforms.items():
                config = self.configs[platform_name]
                
                if config.platform == 'slack':
                    result = platform.test_connection()
                elif config.platform == 'telegram':
                    result = platform.test_connection()
                elif config.platform == 'discord':
                    # Discord doesn't have a simple test method, try getting guilds
                    result = platform.get_guilds()
                    if result['status'] == 'success':
                        result['message'] = 'Discord connection successful'
                else:
                    result = {'status': 'error', 'message': 'Unknown platform type'}
                
                results[platform_name] = result
            
            # Summary
            successful_connections = sum(1 for r in results.values() if r['status'] == 'success')
            total_platforms = len(results)
            
            return {
                'status': 'success',
                'results': results,
                'summary': {
                    'total_platforms': total_platforms,
                    'successful_connections': successful_connections,
                    'failed_connections': total_platforms - successful_connections,
                    'success_rate': (successful_connections / total_platforms) * 100 if total_platforms else 0
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Connection testing failed: {str(e)}'}
    
    def setup_webhook_handler(self, platform_name: str, handler_function: Callable) -> Dict[str, Any]:
        """Setup webhook handler for platform events."""
        try:
            self.webhook_handlers[platform_name] = handler_function
            
            return {
                'status': 'success',
                'platform': platform_name,
                'message': f'Webhook handler set for {platform_name}'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to setup webhook handler: {str(e)}'}
    
    def process_webhook(self, platform_name: str, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming webhook data."""
        try:
            if platform_name not in self.webhook_handlers:
                return {'status': 'error', 'message': f'No webhook handler for {platform_name}'}
            
            handler = self.webhook_handlers[platform_name]
            result = handler(webhook_data)
            
            # Record webhook processing
            self.operation_history.append({
                'type': 'webhook_processed',
                'platform': platform_name,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            return {
                'status': 'success',
                'platform': platform_name,
                'handler_result': result
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Webhook processing failed: {str(e)}'}
    
    def broadcast_message(self, message: str, platforms: List[str] = None, 
                         channel_mapping: Dict[str, str] = None, **kwargs) -> Dict[str, Any]:
        """Broadcast message to all or specified platforms."""
        try:
            target_platforms = platforms or list(self.platforms.keys())
            
            if not channel_mapping:
                return {'status': 'error', 'message': 'Channel mapping required for broadcast'}
            
            return self.send_to_multiple_platforms(target_platforms, channel_mapping, message, **kwargs)
            
        except Exception as e:
            return {'status': 'error', 'message': f'Broadcast failed: {str(e)}'}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all platforms."""
        return {
            'status': 'success',
            'stats': self.stats,
            'configured_platforms': list(self.platforms.keys()),
            'operation_history_count': len(self.operation_history),
            'webhook_handlers_count': len(self.webhook_handlers),
            'available_integrations': {
                'slack': SLACK_SDK_AVAILABLE,
                'discord': DISCORD_PY_AVAILABLE,
                'telegram': TELEGRAM_AVAILABLE
            }
        }
    
    def export_data(self, file_path: str, include_history: bool = True) -> Dict[str, Any]:
        """Export platform data and history."""
        try:
            export_data = {
                'platforms': list(self.platforms.keys()),
                'stats': self.stats,
                'export_timestamp': datetime.now().isoformat()
            }
            
            if include_history:
                export_data['operation_history'] = self.operation_history
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'platforms_exported': len(self.platforms)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Export failed: {str(e)}'}

# Agent framework integration
class MessagingPlatformsAgent:
    """
    Agent wrapper for the messaging platforms tool.
    """
    
    def __init__(self):
        self.tool = MessagingPlatformsTool()
        self.capabilities = [
            'add_platform',
            'send_message',
            'send_to_multiple_platforms',
            'upload_file',
            'get_messages',
            'get_channels',
            'test_connections',
            'broadcast_message',
            'setup_webhook',
            'process_webhook',
            'get_stats',
            'export_data'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific messaging operation."""
        try:
            if action == 'add_platform':
                return self.tool.add_platform(**kwargs)
            elif action == 'send_message':
                return self.tool.send_message(**kwargs)
            elif action == 'send_to_multiple_platforms':
                return self.tool.send_to_multiple_platforms(**kwargs)
            elif action == 'upload_file':
                return self.tool.upload_file(**kwargs)
            elif action == 'get_messages':
                return self.tool.get_messages(**kwargs)
            elif action == 'get_channels':
                return self.tool.get_channels(**kwargs)
            elif action == 'test_connections':
                return self.tool.test_all_connections()
            elif action == 'broadcast_message':
                return self.tool.broadcast_message(**kwargs)
            elif action == 'setup_webhook':
                return self.tool.setup_webhook_handler(**kwargs)
            elif action == 'process_webhook':
                return self.tool.process_webhook(**kwargs)
            elif action == 'get_stats':
                return self.tool.get_comprehensive_stats()
            elif action == 'export_data':
                return self.tool.export_data(**kwargs)
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()

# Quick utility functions
@register_tool(tags=["messenger", "slack", "setup", "configuration"])
def quick_slack_setup(token: str, signing_secret: str = None) -> SlackConfig:
    """Quick Slack configuration setup."""
    return SlackConfig(
        token=token,
        signing_secret=signing_secret,
        api_base_url='https://slack.com/api/'
    )

@register_tool(tags=["messenger", "discord", "setup", "configuration"])
def quick_discord_setup(token: str, guild_id: str = None) -> DiscordConfig:
    """Quick Discord configuration setup."""
    return DiscordConfig(
        token=token,
        guild_id=guild_id,
        intents=['message_content', 'guilds', 'guild_messages']
    )

@register_tool(tags=["messenger", "telegram", "setup", "configuration"])
def quick_telegram_setup(token: str, chat_id: str = None) -> TelegramConfig:
    """Quick Telegram configuration setup."""
    return TelegramConfig(
        token=token,
        chat_id=chat_id,
        parse_mode='HTML'
    )

@register_tool(tags=["messenger", "multi_platform", "agent", "setup"])
def setup_multi_platform_agent(slack_token: str = None, discord_token: str = None, 
                               telegram_token: str = None) -> MessagingPlatformsAgent:
    """Setup agent with multiple platforms."""
    agent = MessagingPlatformsAgent()
    
    if slack_token:
        slack_config = quick_slack_setup(slack_token)
        agent.execute('add_platform', platform_name='slack_main', config=slack_config)
    
    if discord_token:
        discord_config = quick_discord_setup(discord_token)
        agent.execute('add_platform', platform_name='discord_main', config=discord_config)
    
    if telegram_token:
        telegram_config = quick_telegram_setup(telegram_token)
        agent.execute('add_platform', platform_name='telegram_main', config=telegram_config)
    
    return agent

# High-level wrapper functions for quick platform usage
@register_tool(tags=["messenger", "slack", "send", "quick"])
def send_slack_message(token: str, channel: str, message: str, **kwargs: Any) -> Dict[str, Any]:
    """Quick Slack message sending."""
    try:
        config = quick_slack_setup(token)
        slack = SlackIntegration(config)
        return slack.send_message(channel, message, **kwargs)
    except Exception as e:
        return {'status': 'error', 'message': f'Slack message failed: {str(e)}'}

@register_tool(tags=["messenger", "discord", "send", "quick"])
def send_discord_message(token: str, channel_id: str, message: str, **kwargs: Any) -> Dict[str, Any]:
    """Quick Discord message sending."""
    try:
        config = quick_discord_setup(token)
        discord_client = DiscordIntegration(config)
        return discord_client.send_message_sync(channel_id, message, **kwargs)
    except Exception as e:
        return {'status': 'error', 'message': f'Discord message failed: {str(e)}'}

@register_tool(tags=["messenger", "telegram", "send", "quick"])
def send_telegram_message(token: str, chat_id: str, message: str, **kwargs: Any) -> Dict[str, Any]:
    """Quick Telegram message sending."""
    try:
        config = quick_telegram_setup(token, chat_id)
        telegram = TelegramIntegration(config)
        return telegram.send_message(chat_id, message, **kwargs)
    except Exception as e:
        return {'status': 'error', 'message': f'Telegram message failed: {str(e)}'}

@register_tool(tags=["messenger", "multi_platform", "broadcast", "send"])
def broadcast_to_all_platforms(message: str, channel_mapping: Dict[str, str], 
                              slack_token: str = None, discord_token: str = None, 
                              telegram_token: str = None) -> Dict[str, Any]:
    """Broadcast message to multiple platforms."""
    try:
        if not any([slack_token, discord_token, telegram_token]):
            return {'status': 'error', 'message': 'At least one platform token is required'}
        
        agent = setup_multi_platform_agent(slack_token, discord_token, telegram_token)
        
        platforms = []
        if slack_token:
            platforms.append('slack_main')
        if discord_token:
            platforms.append('discord_main')
        if telegram_token:
            platforms.append('telegram_main')
        
        return agent.execute('send_to_multiple_platforms', 
                           platforms=platforms, 
                           channel_mapping=channel_mapping,
                           message=message)
    except Exception as e:
        return {'status': 'error', 'message': f'Broadcast failed: {str(e)}'}

@register_tool(tags=["messenger", "slack", "file", "upload"])
def upload_file_to_slack(token: str, channel: str, file_path: str, 
                        title: str = None, comment: str = None) -> Dict[str, Any]:
    """Upload file to Slack channel."""
    try:
        config = quick_slack_setup(token)
        slack = SlackIntegration(config)
        return slack.upload_file(channel, file_path, title, comment)
    except Exception as e:
        return {'status': 'error', 'message': f'Slack file upload failed: {str(e)}'}

@register_tool(tags=["messenger", "discord", "file", "upload"])
def upload_file_to_discord(token: str, channel_id: str, file_path: str, 
                          content: str = None) -> Dict[str, Any]:
    """Upload file to Discord channel."""
    try:
        config = quick_discord_setup(token)
        discord_client = DiscordIntegration(config)
        return discord_client.upload_file(channel_id, file_path, content)
    except Exception as e:
        return {'status': 'error', 'message': f'Discord file upload failed: {str(e)}'}

@register_tool(tags=["messenger", "telegram", "file", "upload"])
def upload_file_to_telegram(token: str, chat_id: str, file_path: str, 
                           caption: str = None) -> Dict[str, Any]:
    """Upload file to Telegram chat."""
    try:
        config = quick_telegram_setup(token, chat_id)
        telegram = TelegramIntegration(config)
        
        # Determine file type and use appropriate method
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return telegram.send_photo(chat_id, file_path, caption)
        else:
            return telegram.send_document(chat_id, file_path, caption)
    except Exception as e:
        return {'status': 'error', 'message': f'Telegram file upload failed: {str(e)}'}

@register_tool(tags=["messenger", "slack", "channels", "list"])
def get_slack_channels(token: str, types: str = 'public_channel,private_channel') -> Dict[str, Any]:
    """Get list of Slack channels."""
    try:
        config = quick_slack_setup(token)
        slack = SlackIntegration(config)
        return slack.get_channels(types)
    except Exception as e:
        return {'status': 'error', 'message': f'Get Slack channels failed: {str(e)}'}

@register_tool(tags=["messenger", "discord", "channels", "list"])
def get_discord_channels(token: str, guild_id: str = None) -> Dict[str, Any]:
    """Get list of Discord channels."""
    try:
        config = quick_discord_setup(token, guild_id)
        discord_client = DiscordIntegration(config)
        return discord_client.get_channels(guild_id)
    except Exception as e:
        return {'status': 'error', 'message': f'Get Discord channels failed: {str(e)}'}

@register_tool(tags=["messenger", "slack", "messages", "read"])
def get_slack_messages(token: str, channel: str, limit: int = 100, oldest: str = None) -> Dict[str, Any]:
    """Get messages from Slack channel."""
    try:
        config = quick_slack_setup(token)
        slack = SlackIntegration(config)
        return slack.get_messages(channel, limit, oldest)
    except Exception as e:
        return {'status': 'error', 'message': f'Get Slack messages failed: {str(e)}'}

@register_tool(tags=["messenger", "discord", "messages", "read"])
def get_discord_messages(token: str, channel_id: str, limit: int = 50, before: str = None) -> Dict[str, Any]:
    """Get messages from Discord channel."""
    try:
        config = quick_discord_setup(token)
        discord_client = DiscordIntegration(config)
        return discord_client.get_messages(channel_id, limit, before)
    except Exception as e:
        return {'status': 'error', 'message': f'Get Discord messages failed: {str(e)}'}

@register_tool(tags=["messenger", "telegram", "updates", "read"])
def get_telegram_updates(token: str, offset: int = None, limit: int = 100) -> Dict[str, Any]:
    """Get updates from Telegram bot."""
    try:
        config = quick_telegram_setup(token)
        telegram = TelegramIntegration(config)
        return telegram.get_updates(offset, limit)
    except Exception as e:
        return {'status': 'error', 'message': f'Get Telegram updates failed: {str(e)}'}

@register_tool(tags=["messenger", "connection", "test", "all_platforms"])
def test_platform_connections(slack_token: str = None, discord_token: str = None, 
                             telegram_token: str = None) -> Dict[str, Any]:
    """Test connections to all provided platforms."""
    try:
        agent = setup_multi_platform_agent(slack_token, discord_token, telegram_token)
        return agent.execute('test_connections')
    except Exception as e:
        return {'status': 'error', 'message': f'Connection test failed: {str(e)}'}

@register_tool(tags=["messenger", "agent", "create", "advanced"])
def create_messenger_agent() -> MessagingPlatformsAgent:
    """Create a new messaging platforms agent."""
    return MessagingPlatformsAgent()