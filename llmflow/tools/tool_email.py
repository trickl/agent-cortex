"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Email Tool - Manages email operations including sending, receiving, template processing, and attachment handling with multiple provider support.
"""

import smtplib
import imaplib
import poplib
import email
import os
import json
import time
import logging
import hashlib
import tempfile
import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.audio import MIMEAudio
from email.mime.application import MIMEApplication
from email import encoders
from email.header import decode_header, Header
from email.utils import parseaddr, formataddr
import re
import html2text
import ssl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import tool decorator for registration
from .tool_decorator import register_tool

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    GMAIL_API_AVAILABLE = True
except ImportError:
    GMAIL_API_AVAILABLE = False

try:
    import exchangelib
    EXCHANGE_AVAILABLE = True
except ImportError:
    EXCHANGE_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

@dataclass
class EmailConfig:
    
    smtp_server: str = os.getenv('EMAIL_SMTP_SERVER', 'smtp.gmail.com')
    smtp_port: int = int(os.getenv('EMAIL_SMTP_PORT', '587'))
    imap_server: str = os.getenv('EMAIL_IMAP_SERVER', 'imap.gmail.com')
    imap_port: int = int(os.getenv('EMAIL_IMAP_PORT', '993'))
    pop3_server: str = os.getenv('EMAIL_POP3_SERVER', 'pop.gmail.com')
    pop3_port: int = int(os.getenv('EMAIL_POP3_PORT', '995'))
    username: str = os.getenv('EMAIL_USERNAME', '')
    password: str = os.getenv('EMAIL_PASSWORD', '')
    use_tls: bool = os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true'
    use_ssl: bool = os.getenv('EMAIL_USE_SSL', 'false').lower() == 'true'
    timeout: int = 30
    auth_method: str = 'password'  # password, oauth2, app_password

@dataclass
class EmailMessage:
    """Email message data structure."""
    message_id: str
    subject: str
    sender: str
    recipients: List[str]
    cc: List[str]
    bcc: List[str]
    body_text: str
    body_html: str
    attachments: List[Dict[str, Any]]
    headers: Dict[str, str]
    date: str
    folder: str
    is_read: bool
    is_flagged: bool
    size: int

@dataclass
class AttachmentInfo:
    """Attachment information."""
    filename: str
    content_type: str
    size: int
    content: bytes
    content_id: Optional[str] = None

class EmailProcessor:
    """
    Email content processing utilities.
    """
    
    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """Extract plain text from HTML content."""
        try:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            return h.handle(html_content)
        except:
            # Fallback: simple HTML tag removal
            import re
            clean = re.compile('<.*?>')
            return re.sub(clean, '', html_content)
    
    @staticmethod
    def decode_email_header(header_value: str) -> str:
        """Decode email header with proper encoding."""
        try:
            decoded_parts = decode_header(header_value)
            header_text = ''
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        header_text += part.decode(encoding)
                    else:
                        header_text += part.decode('utf-8', errors='ignore')
                else:
                    header_text += part
                    
            return header_text
        except:
            return str(header_value)
    
    @staticmethod
    def extract_email_addresses(text: str) -> List[str]:
        """Extract email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def clean_email_address(email_addr: str) -> str:
        """Clean and validate email address."""
        # Remove name part and keep only email
        _, addr = parseaddr(email_addr)
        return addr.strip().lower()
    
    @staticmethod
    def format_email_address(email: str, name: str = None) -> str:
        """Format email address with optional name."""
        if name:
            return formataddr((name, email))
        return email

class EmailSender:
    """
    Advanced email sending functionality with multiple backends.
    """
    
    def __init__(self, config: EmailConfig = None):
        self.config = config or EmailConfig()
        self.smtp_connection = None
        self.send_history = []
        self.templates = {}
        self.send_stats = {
            'total_sent': 0,
            'total_failed': 0,
            'last_send': None
        }
    
    def connect_smtp(self) -> Dict[str, Any]:
        """Establish SMTP connection."""
        try:
            if self.smtp_connection:
                try:
                    # Test existing connection
                    self.smtp_connection.noop()
                    return {'status': 'success', 'message': 'Already connected'}
                except:
                    self.smtp_connection = None
            
            # Create new connection
            if self.config.use_ssl:
                self.smtp_connection = smtplib.SMTP_SSL(
                    self.config.smtp_server, 
                    self.config.smtp_port,
                    timeout=self.config.timeout
                )
            else:
                self.smtp_connection = smtplib.SMTP(
                    self.config.smtp_server, 
                    self.config.smtp_port,
                    timeout=self.config.timeout
                )
                
                if self.config.use_tls:
                    self.smtp_connection.starttls()
            
            # Authenticate
            if self.config.auth_method == 'password':
                self.smtp_connection.login(self.config.username, self.config.password)
            
            return {
                'status': 'success',
                'message': f'Connected to {self.config.smtp_server}:{self.config.smtp_port}',
                'server': self.config.smtp_server
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'SMTP connection failed: {str(e)}'}
    
    def disconnect_smtp(self) -> Dict[str, Any]:
        """Disconnect SMTP connection."""
        try:
            if self.smtp_connection:
                self.smtp_connection.quit()
                self.smtp_connection = None
            
            return {'status': 'success', 'message': 'SMTP connection closed'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'SMTP disconnect failed: {str(e)}'}
    
    def send_email(self, to_emails: Union[str, List[str]], subject: str, 
                   body: str = None, html_body: str = None,
                   cc_emails: List[str] = None, bcc_emails: List[str] = None,
                   attachments: List[str] = None, reply_to: str = None,
                   from_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Send email with advanced features.
        
        Args:
            to_emails: Recipient email(s)
            subject: Email subject
            body: Plain text body
            html_body: HTML body
            cc_emails: CC recipients
            bcc_emails: BCC recipients
            attachments: List of file paths to attach
            reply_to: Reply-to address
            from_name: Sender name
            **kwargs: Additional parameters
            
        Returns:
            Send result with status and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure connection
            connect_result = self.connect_smtp()
            if connect_result['status'] != 'success':
                return connect_result
            
            # Prepare recipients
            if isinstance(to_emails, str):
                to_emails = [to_emails]
            
            cc_emails = cc_emails or []
            bcc_emails = bcc_emails or []
            
            # Validate email addresses
            all_recipients = to_emails + cc_emails + bcc_emails
            for email_addr in all_recipients:
                if not self._validate_email(email_addr):
                    return {'status': 'error', 'message': f'Invalid email address: {email_addr}'}
            
            # Create message
            msg = MIMEMultipart('alternative')
            
            # Set headers
            msg['From'] = EmailProcessor.format_email_address(self.config.username, from_name)
            msg['To'] = ', '.join(to_emails)
            if cc_emails:
                msg['Cc'] = ', '.join(cc_emails)
            msg['Subject'] = Header(subject, 'utf-8')
            msg['Date'] = email.utils.formatdate(localtime=True)
            
            if reply_to:
                msg['Reply-To'] = reply_to
            
            # Add custom headers
            for key, value in kwargs.get('headers', {}).items():
                msg[key] = value
            
            # Add message ID for tracking
            message_id = email.utils.make_msgid()
            msg['Message-ID'] = message_id
            
            # Add body content
            if body:
                text_part = MIMEText(body, 'plain', 'utf-8')
                msg.attach(text_part)
            
            if html_body:
                html_part = MIMEText(html_body, 'html', 'utf-8')
                msg.attach(html_part)
            
            if not body and not html_body:
                # Default empty body
                msg.attach(MIMEText('', 'plain', 'utf-8'))
            
            # Add attachments
            attachment_info = []
            if attachments:
                for file_path in attachments:
                    attach_result = self._attach_file(msg, file_path)
                    if attach_result['status'] == 'success':
                        attachment_info.append(attach_result['info'])
                    else:
                        logging.warning(f"Failed to attach {file_path}: {attach_result['message']}")
            
            # Send email
            all_recipients = to_emails + cc_emails + bcc_emails
            self.smtp_connection.send_message(msg, to_addrs=all_recipients)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.send_stats['total_sent'] += 1
            self.send_stats['last_send'] = datetime.now().isoformat()
            
            # Record in history
            send_record = {
                'message_id': message_id,
                'timestamp': datetime.now().isoformat(),
                'to_emails': to_emails,
                'cc_emails': cc_emails,
                'bcc_emails': bcc_emails,
                'subject': subject,
                'attachments_count': len(attachment_info),
                'processing_time': processing_time,
                'success': True
            }
            self.send_history.append(send_record)
            
            return {
                'status': 'success',
                'message_id': message_id,
                'recipients_count': len(all_recipients),
                'attachments_count': len(attachment_info),
                'processing_time': processing_time,
                'message': f'Email sent successfully to {len(all_recipients)} recipients'
            }
            
        except Exception as e:
            # Update statistics
            self.send_stats['total_failed'] += 1
            
            # Record failed attempt
            send_record = {
                'timestamp': datetime.now().isoformat(),
                'to_emails': to_emails if 'to_emails' in locals() else [],
                'subject': subject,
                'error': str(e),
                'success': False
            }
            self.send_history.append(send_record)
            
            return {'status': 'error', 'message': f'Failed to send email: {str(e)}'}
    
    def _validate_email(self, email_addr: str) -> bool:
        """Validate email address format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email_addr.strip()) is not None
    
    def _attach_file(self, msg: MIMEMultipart, file_path: str) -> Dict[str, Any]:
        """Attach file to email message."""
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # Check file size (limit to 25MB)
            if file_size > 25 * 1024 * 1024:
                return {'status': 'error', 'message': f'File too large: {filename} ({file_size} bytes)'}
            
            # Determine MIME type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            main_type, sub_type = content_type.split('/', 1)
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            # Create appropriate MIME object
            if main_type == 'text':
                attachment = MIMEText(file_data.decode('utf-8'), sub_type)
            elif main_type == 'image':
                attachment = MIMEImage(file_data, sub_type)
            elif main_type == 'audio':
                attachment = MIMEAudio(file_data, sub_type)
            else:
                attachment = MIMEApplication(file_data, sub_type)
            
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(attachment)
            
            return {
                'status': 'success',
                'info': {
                    'filename': filename,
                    'content_type': content_type,
                    'size': file_size
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to attach file: {str(e)}'}
    
    def send_bulk_email(self, recipients: List[Dict[str, Any]], subject_template: str,
                       body_template: str = None, html_template: str = None,
                       **kwargs) -> Dict[str, Any]:
        """
        Send bulk emails with personalization.
        
        Args:
            recipients: List of recipient dictionaries with email and variables
            subject_template: Subject template with placeholders
            body_template: Body template with placeholders
            html_template: HTML template with placeholders
            **kwargs: Additional parameters
            
        Returns:
            Bulk send results
        """
        try:
            results = []
            successful_sends = 0
            failed_sends = 0
            
            for i, recipient in enumerate(recipients):
                try:
                    # Extract recipient info
                    to_email = recipient.get('email')
                    variables = recipient.get('variables', {})
                    
                    if not to_email:
                        results.append({
                            'index': i,
                            'email': 'unknown',
                            'status': 'error',
                            'message': 'No email address provided'
                        })
                        failed_sends += 1
                        continue
                    
                    # Render templates
                    subject = self._render_template(subject_template, variables)
                    body = self._render_template(body_template, variables) if body_template else None
                    html_body = self._render_template(html_template, variables) if html_template else None
                    
                    # Send email
                    send_result = self.send_email(
                        to_emails=to_email,
                        subject=subject,
                        body=body,
                        html_body=html_body,
                        **kwargs
                    )
                    
                    results.append({
                        'index': i,
                        'email': to_email,
                        'status': send_result['status'],
                        'message_id': send_result.get('message_id'),
                        'message': send_result.get('message', ''),
                        'processing_time': send_result.get('processing_time', 0)
                    })
                    
                    if send_result['status'] == 'success':
                        successful_sends += 1
                    else:
                        failed_sends += 1
                        
                except Exception as e:
                    results.append({
                        'index': i,
                        'email': recipient.get('email', 'unknown'),
                        'status': 'error',
                        'message': str(e)
                    })
                    failed_sends += 1
            
            return {
                'status': 'success',
                'total_recipients': len(recipients),
                'successful_sends': successful_sends,
                'failed_sends': failed_sends,
                'success_rate': (successful_sends / len(recipients)) * 100 if recipients else 0,
                'results': results
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Bulk email sending failed: {str(e)}'}
    
    def _render_template(self, template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables."""
        if not template:
            return ''
        
        try:
            if JINJA2_AVAILABLE:
                # Use Jinja2 for advanced templating
                jinja_template = Template(template)
                return jinja_template.render(**variables)
            else:
                # Simple string formatting
                return template.format(**variables)
        except Exception as e:
            logging.warning(f"Template rendering failed: {e}")
            return template
    
    def save_template(self, name: str, subject: str, body: str = None, 
                     html_body: str = None) -> Dict[str, Any]:
        """Save email template."""
        try:
            self.templates[name] = {
                'subject': subject,
                'body': body,
                'html_body': html_body,
                'created_at': datetime.now().isoformat()
            }
            
            return {
                'status': 'success',
                'template_name': name,
                'message': f'Template "{name}" saved successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to save template: {str(e)}'}
    
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get saved email template."""
        if name not in self.templates:
            return {'status': 'error', 'message': f'Template "{name}" not found'}
        
        return {
            'status': 'success',
            'template': self.templates[name]
        }
    
    def list_templates(self) -> Dict[str, Any]:
        """List all saved templates."""
        return {
            'status': 'success',
            'templates': list(self.templates.keys()),
            'total_templates': len(self.templates)
        }
    
    def get_send_stats(self) -> Dict[str, Any]:
        """Get sending statistics."""
        return {
            'status': 'success',
            'stats': self.send_stats,
            'history_count': len(self.send_history)
        }

class EmailReader:
    """
    Advanced email reading functionality with multiple protocols.
    """
    
    def __init__(self, config: EmailConfig = None):
        self.config = config or EmailConfig()
        self.imap_connection = None
        self.pop3_connection = None
        self.current_folder = 'INBOX'
        self.read_history = []
        self.message_cache = {}
    
    def connect_imap(self) -> Dict[str, Any]:
        """Establish IMAP connection."""
        try:
            if self.imap_connection:
                try:
                    # Test existing connection
                    self.imap_connection.noop()
                    return {'status': 'success', 'message': 'Already connected'}
                except:
                    self.imap_connection = None
            
            # Create new connection
            if self.config.use_ssl:
                self.imap_connection = imaplib.IMAP4_SSL(
                    self.config.imap_server,
                    self.config.imap_port
                )
            else:
                self.imap_connection = imaplib.IMAP4(
                    self.config.imap_server,
                    self.config.imap_port
                )
                
                if self.config.use_tls:
                    self.imap_connection.starttls()
            
            # Authenticate
            self.imap_connection.login(self.config.username, self.config.password)
            
            return {
                'status': 'success',
                'message': f'Connected to {self.config.imap_server}:{self.config.imap_port}',
                'server': self.config.imap_server
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'IMAP connection failed: {str(e)}'}
    
    def disconnect_imap(self) -> Dict[str, Any]:
        """Disconnect IMAP connection."""
        try:
            if self.imap_connection:
                self.imap_connection.close()
                self.imap_connection.logout()
                self.imap_connection = None
            
            return {'status': 'success', 'message': 'IMAP connection closed'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'IMAP disconnect failed: {str(e)}'}
    
    def list_folders(self) -> Dict[str, Any]:
        """List available email folders."""
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            status, folders = self.imap_connection.list()
            
            if status != 'OK':
                return {'status': 'error', 'message': 'Failed to list folders'}
            
            folder_list = []
            for folder in folders:
                # Parse folder info
                parts = folder.decode('utf-8').split('"')
                if len(parts) >= 3:
                    folder_name = parts[-2]
                    folder_list.append(folder_name)
            
            return {
                'status': 'success',
                'folders': folder_list,
                'total_folders': len(folder_list),
                'current_folder': self.current_folder
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to list folders: {str(e)}'}
    
    def select_folder(self, folder_name: str) -> Dict[str, Any]:
        """Select email folder."""
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            status, data = self.imap_connection.select(folder_name)
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to select folder: {folder_name}'}
            
            self.current_folder = folder_name
            message_count = int(data[0])
            
            return {
                'status': 'success',
                'folder': folder_name,
                'message_count': message_count,
                'message': f'Selected folder "{folder_name}" with {message_count} messages'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to select folder: {str(e)}'}
    
    def search_emails(self, criteria: str = 'ALL', limit: int = 50) -> Dict[str, Any]:
        """
        Search emails with IMAP search criteria.
        
        Args:
            criteria: IMAP search criteria (e.g., 'ALL', 'UNSEEN', 'FROM "sender@example.com"')
            limit: Maximum number of messages to return
            
        Returns:
            Search results with message UIDs
        """
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            # Ensure folder is selected
            if not self.current_folder:
                folder_result = self.select_folder('INBOX')
                if folder_result['status'] != 'success':
                    return folder_result
            
            # Search for messages
            status, data = self.imap_connection.search(None, criteria)
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Search failed for criteria: {criteria}'}
            
            message_uids = data[0].split()
            
            # Apply limit
            if limit and len(message_uids) > limit:
                message_uids = message_uids[-limit:]  # Get most recent
            
            return {
                'status': 'success',
                'message_uids': [uid.decode('utf-8') for uid in message_uids],
                'total_found': len(message_uids),
                'criteria': criteria,
                'folder': self.current_folder
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Email search failed: {str(e)}'}
    
    def fetch_email(self, message_uid: str, include_body: bool = True, 
                   include_attachments: bool = False) -> Dict[str, Any]:
        """
        Fetch email message by UID.
        
        Args:
            message_uid: Message UID
            include_body: Whether to include message body
            include_attachments: Whether to include attachments
            
        Returns:
            Email message data
        """
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            # Check cache first
            cache_key = f"{self.current_folder}:{message_uid}"
            if cache_key in self.message_cache and not include_attachments:
                return {
                    'status': 'success',
                    'message': self.message_cache[cache_key],
                    'from_cache': True
                }
            
            # Fetch message
            status, data = self.imap_connection.fetch(message_uid, '(RFC822)')
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to fetch message: {message_uid}'}
            
            raw_email = data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Parse email
            parsed_message = self._parse_email_message(
                email_message, 
                include_body=include_body,
                include_attachments=include_attachments
            )
            
            # Add UID and folder info
            parsed_message.message_id = message_uid
            parsed_message.folder = self.current_folder
            
            # Cache message (without attachments)
            if not include_attachments:
                self.message_cache[cache_key] = parsed_message
            
            # Record in history
            self.read_history.append({
                'timestamp': datetime.now().isoformat(),
                'message_uid': message_uid,
                'folder': self.current_folder,
                'subject': parsed_message.subject
            })
            
            return {
                'status': 'success',
                'message': asdict(parsed_message),
                'from_cache': False
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to fetch email: {str(e)}'}
    
    def fetch_emails(self, message_uids: List[str], include_body: bool = True,
                    include_attachments: bool = False) -> Dict[str, Any]:
        """Fetch multiple emails."""
        try:
            messages = []
            successful_fetches = 0
            failed_fetches = 0
            
            for uid in message_uids:
                result = self.fetch_email(uid, include_body, include_attachments)
                
                if result['status'] == 'success':
                    messages.append(result['message'])
                    successful_fetches += 1
                else:
                    failed_fetches += 1
                    logging.warning(f"Failed to fetch message {uid}: {result['message']}")
            
            return {
                'status': 'success',
                'messages': messages,
                'total_requested': len(message_uids),
                'successful_fetches': successful_fetches,
                'failed_fetches': failed_fetches
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to fetch emails: {str(e)}'}
    
    def _parse_email_message(self, email_message: email.message.Message,
                           include_body: bool = True, 
                           include_attachments: bool = False) -> EmailMessage:
        """Parse email.message.Message into EmailMessage dataclass."""
        
        # Extract headers
        subject = EmailProcessor.decode_email_header(email_message.get('Subject', ''))
        sender = EmailProcessor.decode_email_header(email_message.get('From', ''))
        date_str = email_message.get('Date', '')
        message_id = email_message.get('Message-ID', '')
        
        # Parse recipients
        to_header = email_message.get('To', '')
        cc_header = email_message.get('Cc', '')
        bcc_header = email_message.get('Bcc', '')
        
        recipients = EmailProcessor.extract_email_addresses(to_header)
        cc_recipients = EmailProcessor.extract_email_addresses(cc_header)
        bcc_recipients = EmailProcessor.extract_email_addresses(bcc_header)
        
        # Extract body and attachments
        body_text = ''
        body_html = ''
        attachments = []
        
        if include_body or include_attachments:
            if email_message.is_multipart():
                for part in email_message.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get('Content-Disposition', ''))
                    
                    if 'attachment' in content_disposition and include_attachments:
                        # Handle attachment
                        attachment = self._extract_attachment(part)
                        if attachment:
                            attachments.append(asdict(attachment))
                    
                    elif include_body:
                        if content_type == 'text/plain' and 'attachment' not in content_disposition:
                            charset = part.get_content_charset() or 'utf-8'
                            body_text += part.get_payload(decode=True).decode(charset, errors='ignore')
                        
                        elif content_type == 'text/html' and 'attachment' not in content_disposition:
                            charset = part.get_content_charset() or 'utf-8'
                            body_html += part.get_payload(decode=True).decode(charset, errors='ignore')
            else:
                # Non-multipart message
                if include_body:
                    content_type = email_message.get_content_type()
                    charset = email_message.get_content_charset() or 'utf-8'
                    content = email_message.get_payload(decode=True).decode(charset, errors='ignore')
                    
                    if content_type == 'text/html':
                        body_html = content
                        body_text = EmailProcessor.extract_text_from_html(content)
                    else:
                        body_text = content
        
        # Extract all headers
        headers = {}
        for key, value in email_message.items():
            headers[key] = EmailProcessor.decode_email_header(value)
        
        return EmailMessage(
            message_id=message_id,
            subject=subject,
            sender=sender,
            recipients=recipients,
            cc=cc_recipients,
            bcc=bcc_recipients,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            headers=headers,
            date=date_str,
            folder='',  # Will be set by caller
            is_read=True,  # Assume read when fetched
            is_flagged=False,
            size=len(str(email_message))
        )
    
    def _extract_attachment(self, part: email.message.Message) -> Optional[AttachmentInfo]:
        """Extract attachment from email part."""
        try:
            filename = part.get_filename()
            if not filename:
                return None
            
            # Decode filename
            filename = EmailProcessor.decode_email_header(filename)
            
            content_type = part.get_content_type()
            content = part.get_payload(decode=True)
            content_id = part.get('Content-ID')
            
            return AttachmentInfo(
                filename=filename,
                content_type=content_type,
                size=len(content),
                content=content,
                content_id=content_id
            )
            
        except Exception as e:
            logging.warning(f"Failed to extract attachment: {e}")
            return None
    
    def mark_as_read(self, message_uid: str) -> Dict[str, Any]:
        """Mark message as read."""
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            status, data = self.imap_connection.store(message_uid, '+FLAGS', '\\Seen')
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to mark message as read: {message_uid}'}
            
            return {
                'status': 'success',
                'message_uid': message_uid,
                'message': 'Message marked as read'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to mark as read: {str(e)}'}
    
    def mark_as_unread(self, message_uid: str) -> Dict[str, Any]:
        """Mark message as unread."""
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            status, data = self.imap_connection.store(message_uid, '-FLAGS', '\\Seen')
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to mark message as unread: {message_uid}'}
            
            return {
                'status': 'success',
                'message_uid': message_uid,
                'message': 'Message marked as unread'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to mark as unread: {str(e)}'}
    
    def delete_email(self, message_uid: str) -> Dict[str, Any]:
        """Delete email message."""
        try:
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            # Mark for deletion
            status, data = self.imap_connection.store(message_uid, '+FLAGS', '\\Deleted')
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to mark message for deletion: {message_uid}'}
            
            # Expunge to permanently delete
            self.imap_connection.expunge()
            
            return {
                'status': 'success',
                'message_uid': message_uid,
                'message': 'Message deleted successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete message: {str(e)}'}
    
    def get_folder_info(self, folder_name: str = None) -> Dict[str, Any]:
        """Get folder information and statistics."""
        try:
            if folder_name:
                select_result = self.select_folder(folder_name)
                if select_result['status'] != 'success':
                    return select_result
            
            connect_result = self.connect_imap()
            if connect_result['status'] != 'success':
                return connect_result
            
            # Get folder status
            status, data = self.imap_connection.status(self.current_folder, '(MESSAGES RECENT UNSEEN)')
            
            if status != 'OK':
                return {'status': 'error', 'message': f'Failed to get folder status: {self.current_folder}'}
            
            # Parse status response
            status_info = data[0].decode('utf-8')
            
            # Extract counts using regex
            messages_match = re.search(r'MESSAGES (\d+)', status_info)
            recent_match = re.search(r'RECENT (\d+)', status_info)
            unseen_match = re.search(r'UNSEEN (\d+)', status_info)
            
            return {
                'status': 'success',
                'folder': self.current_folder,
                'total_messages': int(messages_match.group(1)) if messages_match else 0,
                'recent_messages': int(recent_match.group(1)) if recent_match else 0,
                'unread_messages': int(unseen_match.group(1)) if unseen_match else 0,
                'status_info': status_info
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get folder info: {str(e)}'}

class EmailTool:
    """
    Comprehensive email tool combining sending and reading functionality.
    """
    
    def __init__(self, config: EmailConfig = None):
        self.config = config or EmailConfig()
        self.sender = EmailSender(self.config)
        self.reader = EmailReader(self.config)
        
        # Unified statistics
        self.tool_stats = {
            'total_operations': 0,
            'last_operation': None,
            'errors_count': 0
        }
        
        # Auto-responder settings
        self.auto_responder_enabled = False
        self.auto_responder_template = None
        self.auto_responder_conditions = []
    
    def configure(self, **config_params) -> Dict[str, Any]:
        """Update email configuration."""
        try:
            for key, value in config_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
            # Update sender and reader configs
            self.sender.config = self.config
            self.reader.config = self.config
            
            return {
                'status': 'success',
                'message': 'Configuration updated',
                'config': asdict(self.config)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Configuration failed: {str(e)}'}
    
    def send_email(self, **kwargs) -> Dict[str, Any]:
        """Send email using sender."""
        self.tool_stats['total_operations'] += 1
        self.tool_stats['last_operation'] = 'send_email'
        
        result = self.sender.send_email(**kwargs)
        
        if result['status'] == 'error':
            self.tool_stats['errors_count'] += 1
        
        return result
    
    def read_emails(self, folder: str = 'INBOX', limit: int = 10, 
                   criteria: str = 'ALL', include_body: bool = True) -> Dict[str, Any]:
        """Read emails from specified folder."""
        try:
            self.tool_stats['total_operations'] += 1
            self.tool_stats['last_operation'] = 'read_emails'
            
            # Select folder
            folder_result = self.reader.select_folder(folder)
            if folder_result['status'] != 'success':
                return folder_result
            
            # Search for messages
            search_result = self.reader.search_emails(criteria, limit)
            if search_result['status'] != 'success':
                return search_result
            
            # Fetch messages
            if search_result['message_uids']:
                fetch_result = self.reader.fetch_emails(
                    search_result['message_uids'],
                    include_body=include_body
                )
                
                return {
                    'status': 'success',
                    'folder': folder,
                    'criteria': criteria,
                    'messages': fetch_result['messages'],
                    'total_found': len(fetch_result['messages'])
                }
            else:
                return {
                    'status': 'success',
                    'folder': folder,
                    'criteria': criteria,
                    'messages': [],
                    'total_found': 0
                }
                
        except Exception as e:
            self.tool_stats['errors_count'] += 1
            return {'status': 'error', 'message': f'Failed to read emails: {str(e)}'}
    
    def save_attachment(self, message_uid: str, attachment_filename: str, 
                       save_path: str) -> Dict[str, Any]:
        """Save specific attachment from email."""
        try:
            # Fetch email with attachments
            email_result = self.reader.fetch_email(message_uid, include_attachments=True)
            if email_result['status'] != 'success':
                return email_result
            
            message = email_result['message']
            
            # Find attachment
            for attachment in message['attachments']:
                if attachment['filename'] == attachment_filename:
                    # Save attachment
                    with open(save_path, 'wb') as f:
                        f.write(base64.b64decode(attachment['content']))
                    
                    return {
                        'status': 'success',
                        'filename': attachment_filename,
                        'save_path': save_path,
                        'size': attachment['size'],
                        'message': f'Attachment saved: {attachment_filename}'
                    }
            
            return {'status': 'error', 'message': f'Attachment not found: {attachment_filename}'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to save attachment: {str(e)}'}
    
    def setup_auto_responder(self, template: str, conditions: List[str] = None,
                           enabled: bool = True) -> Dict[str, Any]:
        """Setup automatic email responder."""
        try:
            self.auto_responder_template = template
            self.auto_responder_conditions = conditions or []
            self.auto_responder_enabled = enabled
            
            return {
                'status': 'success',
                'enabled': enabled,
                'template': template,
                'conditions': conditions,
                'message': f'Auto-responder {"enabled" if enabled else "disabled"}'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to setup auto-responder: {str(e)}'}
    
    def process_auto_responses(self) -> Dict[str, Any]:
        """Process emails for auto-responses."""
        if not self.auto_responder_enabled:
            return {'status': 'info', 'message': 'Auto-responder is disabled'}
        
        try:
            # Search for unread emails
            unread_result = self.read_emails(criteria='UNSEEN', include_body=False)
            
            if unread_result['status'] != 'success':
                return unread_result
            
            responses_sent = 0
            
            for message in unread_result['messages']:
                # Check conditions
                should_respond = True
                for condition in self.auto_responder_conditions:
                    if condition.lower() not in message['subject'].lower():
                        should_respond = False
                        break
                
                if should_respond:
                    # Send auto-response
                    response_result = self.sender.send_email(
                        to_emails=message['sender'],
                        subject=f"Re: {message['subject']}",
                        body=self.auto_responder_template
                    )
                    
                    if response_result['status'] == 'success':
                        responses_sent += 1
            
            return {
                'status': 'success',
                'responses_sent': responses_sent,
                'total_unread': len(unread_result['messages'])
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Auto-response processing failed: {str(e)}'}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'status': 'success',
            'tool_stats': self.tool_stats,
            'sender_stats': self.sender.get_send_stats()['stats'],
            'reader_history_count': len(self.reader.read_history),
            'auto_responder': {
                'enabled': self.auto_responder_enabled,
                'has_template': self.auto_responder_template is not None,
                'conditions_count': len(self.auto_responder_conditions)
            },
            'config': asdict(self.config)
        }
    
    def export_emails(self, folder: str = 'INBOX', format: str = 'json',
                     file_path: str = None, limit: int = 100) -> Dict[str, Any]:
        """Export emails to file."""
        try:
            # Read emails
            emails_result = self.read_emails(folder=folder, limit=limit, include_body=True)
            if emails_result['status'] != 'success':
                return emails_result
            
            # Generate filename if not provided
            if not file_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = f'emails_export_{folder}_{timestamp}.{format}'
            
            # Export based on format
            if format == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(emails_result['messages'], f, indent=2, ensure_ascii=False)
            
            elif format == 'csv' and PANDAS_AVAILABLE:
                # Flatten email data for CSV
                flattened_data = []
                for msg in emails_result['messages']:
                    flattened_data.append({
                        'subject': msg['subject'],
                        'sender': msg['sender'],
                        'date': msg['date'],
                        'body_text': msg['body_text'][:500] + '...' if len(msg['body_text']) > 500 else msg['body_text'],
                        'recipients_count': len(msg['recipients']),
                        'attachments_count': len(msg['attachments'])
                    })
                
                df = pd.DataFrame(flattened_data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            else:
                return {'status': 'error', 'message': f'Unsupported format: {format}'}
            
            file_size = os.path.getsize(file_path)
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': format,
                'emails_exported': len(emails_result['messages']),
                'file_size': file_size,
                'folder': folder
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Export failed: {str(e)}'}


# Agent framework integration
class EmailAgent:
    """
    Agent wrapper for the email tool.
    """
    
    def __init__(self, config: EmailConfig = None):
        self.tool = EmailTool(config)
        self.capabilities = [
            'send_email',
            'send_bulk_email',
            'read_emails',
            'search_emails',
            'fetch_email',
            'save_attachment',
            'mark_read',
            'mark_unread',
            'delete_email',
            'list_folders',
            'get_folder_info',
            'save_template',
            'get_template',
            'setup_auto_responder',
            'export_emails',
            'configure',
            'get_stats'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific email operation."""
        try:
            if action == 'send_email':
                return self.tool.send_email(**kwargs)
            elif action == 'send_bulk_email':
                return self.tool.sender.send_bulk_email(**kwargs)
            elif action == 'read_emails':
                return self.tool.read_emails(**kwargs)
            elif action == 'search_emails':
                return self.tool.reader.search_emails(**kwargs)
            elif action == 'fetch_email':
                return self.tool.reader.fetch_email(**kwargs)
            elif action == 'save_attachment':
                return self.tool.save_attachment(**kwargs)
            elif action == 'mark_read':
                return self.tool.reader.mark_as_read(**kwargs)
            elif action == 'mark_unread':
                return self.tool.reader.mark_as_unread(**kwargs)
            elif action == 'delete_email':
                return self.tool.reader.delete_email(**kwargs)
            elif action == 'list_folders':
                return self.tool.reader.list_folders()
            elif action == 'get_folder_info':
                return self.tool.reader.get_folder_info(**kwargs)
            elif action == 'save_template':
                return self.tool.sender.save_template(**kwargs)
            elif action == 'get_template':
                return self.tool.sender.get_template(**kwargs)
            elif action == 'list_templates':
                return self.tool.sender.list_templates()
            elif action == 'setup_auto_responder':
                return self.tool.setup_auto_responder(**kwargs)
            elif action == 'process_auto_responses':
                return self.tool.process_auto_responses()
            elif action == 'export_emails':
                return self.tool.export_emails(**kwargs)
            elif action == 'configure':
                return self.tool.configure(**kwargs)
            elif action == 'get_stats':
                return self.tool.get_comprehensive_stats()
            elif action == 'connect_smtp':
                return self.tool.sender.connect_smtp()
            elif action == 'connect_imap':
                return self.tool.reader.connect_imap()
            elif action == 'disconnect':
                smtp_result = self.tool.sender.disconnect_smtp()
                imap_result = self.tool.reader.disconnect_imap()
                return {'status': 'success', 'smtp': smtp_result, 'imap': imap_result}
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()


# Quick utility functions
@register_tool(tags=["email", "communication", "send", "quick"])
def quick_send_email(to_email: str, subject: str, body: str, 
                    smtp_server: str = 'smtp.gmail.com', username: str = '', 
                    password: str = '') -> Dict[str, Any]:
    """Quick email sending function."""
    config = EmailConfig(
        smtp_server=smtp_server,
        username=username,
        password=password
    )
    
    tool = EmailTool(config)
    return tool.send_email(to_emails=to_email, subject=subject, body=body)

@register_tool(tags=["email", "communication", "read", "quick"])
def quick_read_emails(folder: str = 'INBOX', limit: int = 5,
                     imap_server: str = 'imap.gmail.com', username: str = '',
                     password: str = '') -> Dict[str, Any]:
    """Quick email reading function."""
    config = EmailConfig(
        imap_server=imap_server,
        username=username,
        password=password
    )
    
    tool = EmailTool(config)
    return tool.read_emails(folder=folder, limit=limit)

@register_tool(tags=["email", "configuration", "gmail", "setup"])
def setup_gmail_config(username: str, password: str) -> EmailConfig:
    """Setup Gmail configuration."""
    return EmailConfig(
        smtp_server='smtp.gmail.com',
        smtp_port=587,
        imap_server='imap.gmail.com',
        imap_port=993,
        username=username,
        password=password,
        use_tls=True,
        use_ssl=False
    )

@register_tool(tags=["email", "configuration", "outlook", "setup"])
def setup_outlook_config(username: str, password: str) -> EmailConfig:
    """Setup Outlook/Hotmail configuration."""
    return EmailConfig(
        smtp_server='smtp.live.com',
        smtp_port=587,
        imap_server='outlook.office365.com',
        imap_port=993,
        username=username,
        password=password,
        use_tls=True,
        use_ssl=False
    )

# Advanced email functions for complex operations
@register_tool(tags=["email", "communication", "send", "advanced"])
def send_email_advanced(to_emails: Union[str, List[str]], subject: str, 
                       body: str = None, html_body: str = None,
                       cc_emails: Optional[List[str]] = None, 
                       bcc_emails: Optional[List[str]] = None,
                       attachments: Optional[List[str]] = None,
                       smtp_server: str = 'smtp.gmail.com', username: str = '', 
                       password: str = '', from_name: str = None) -> Dict[str, Any]:
    """
    Send advanced email with full features.
    
    Args:
        to_emails: Recipients (string or list)
        subject: Email subject
        body: Plain text body
        html_body: HTML body
        cc_emails: CC recipients
        bcc_emails: BCC recipients  
        attachments: List of file paths to attach
        smtp_server: SMTP server address
        username: Email username
        password: Email password
        from_name: Display name for sender
        
    Returns:
        Dictionary with send status and details
    """
    try:
        config = EmailConfig(
            smtp_server=smtp_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.send_email(
            to_emails=to_emails,
            subject=subject,
            body=body,
            html_body=html_body,
            cc_emails=cc_emails,
            bcc_emails=bcc_emails,
            attachments=attachments,
            from_name=from_name
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to send email: {str(e)}'}

@register_tool(tags=["email", "communication", "read", "advanced"])
def read_emails_advanced(folder: str = 'INBOX', limit: int = 10, 
                        criteria: str = 'ALL', include_body: bool = True,
                        imap_server: str = 'imap.gmail.com', username: str = '',
                        password: str = '') -> Dict[str, Any]:
    """
    Read emails with advanced filtering and options.
    
    Args:
        folder: Email folder to read from
        limit: Maximum number of emails to fetch
        criteria: Search criteria (ALL, UNSEEN, FROM, SUBJECT, etc.)
        include_body: Whether to include email body content
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with emails and metadata
    """
    try:
        config = EmailConfig(
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.read_emails(
            folder=folder,
            limit=limit,
            criteria=criteria,
            include_body=include_body
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to read emails: {str(e)}'}

@register_tool(tags=["email", "communication", "bulk", "marketing"])
def send_bulk_emails(recipients: List[Dict[str, Any]], subject_template: str,
                    body_template: str = None, html_template: str = None,
                    smtp_server: str = 'smtp.gmail.com', username: str = '',
                    password: str = '') -> Dict[str, Any]:
    """
    Send bulk emails with personalization.
    
    Args:
        recipients: List of recipient data with email and template variables
        subject_template: Subject template with {{variable}} placeholders
        body_template: Plain text body template
        html_template: HTML body template
        smtp_server: SMTP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with bulk send results
    """
    try:
        config = EmailConfig(
            smtp_server=smtp_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.sender.send_bulk_email(
            recipients=recipients,
            subject_template=subject_template,
            body_template=body_template,
            html_template=html_template
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to send bulk emails: {str(e)}'}

@register_tool(tags=["email", "search", "advanced", "filter"])
def search_emails(criteria: str = 'ALL', limit: int = 50, folder: str = 'INBOX',
                 imap_server: str = 'imap.gmail.com', username: str = '',
                 password: str = '') -> Dict[str, Any]:
    """
    Search emails with specific criteria.
    
    Args:
        criteria: IMAP search criteria (e.g., 'FROM "example@email.com"', 'SUBJECT "test"')
        limit: Maximum number of results
        folder: Folder to search in
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with search results
    """
    try:
        config = EmailConfig(
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        tool.reader.connect_imap()
        tool.reader.select_folder(folder)
        result = tool.reader.search_emails(criteria=criteria, limit=limit)
        tool.reader.disconnect_imap()
        
        return result
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to search emails: {str(e)}'}

@register_tool(tags=["email", "attachments", "download", "save"])
def save_email_attachment(message_uid: str, attachment_filename: str, 
                         save_path: str, folder: str = 'INBOX',
                         imap_server: str = 'imap.gmail.com', username: str = '',
                         password: str = '') -> Dict[str, Any]:
    """
    Save an email attachment to file.
    
    Args:
        message_uid: Unique identifier of the email message
        attachment_filename: Name of the attachment to save
        save_path: Local path where to save the attachment
        folder: Email folder containing the message
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with save operation status
    """
    try:
        config = EmailConfig(
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.save_attachment(
            message_uid=message_uid,
            attachment_filename=attachment_filename,
            save_path=save_path
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to save attachment: {str(e)}'}

@register_tool(tags=["email", "template", "automation", "save"])
def save_email_template(name: str, subject: str, body: str = None, 
                       html_body: str = None, smtp_server: str = 'smtp.gmail.com',
                       username: str = '', password: str = '') -> Dict[str, Any]:
    """
    Save an email template for reuse.
    
    Args:
        name: Template name
        subject: Email subject template
        body: Plain text body template
        html_body: HTML body template
        smtp_server: SMTP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with save status
    """
    try:
        config = EmailConfig(
            smtp_server=smtp_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.sender.save_template(
            name=name,
            subject=subject,
            body=body,
            html_body=html_body
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to save template: {str(e)}'}

@register_tool(tags=["email", "folders", "list", "management"])
def list_email_folders(imap_server: str = 'imap.gmail.com', username: str = '',
                      password: str = '') -> Dict[str, Any]:
    """
    List all available email folders.
    
    Args:
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with folder list
    """
    try:
        config = EmailConfig(
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        tool.reader.connect_imap()
        result = tool.reader.list_folders()
        tool.reader.disconnect_imap()
        
        return result
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to list folders: {str(e)}'}

@register_tool(tags=["email", "export", "backup", "data"])
def export_emails_to_file(folder: str = 'INBOX', format: str = 'json',
                         file_path: str = None, limit: int = 100,
                         imap_server: str = 'imap.gmail.com', username: str = '',
                         password: str = '') -> Dict[str, Any]:
    """
    Export emails to a file for backup or analysis.
    
    Args:
        folder: Email folder to export
        format: Export format ('json' or 'csv')
        file_path: Output file path (auto-generated if not provided)
        limit: Maximum number of emails to export
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with export results
    """
    try:
        config = EmailConfig(
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.export_emails(
            folder=folder,
            format=format,
            file_path=file_path,
            limit=limit
        )
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to export emails: {str(e)}'}

@register_tool(tags=["email", "stats", "analytics", "info"])
def get_email_stats(smtp_server: str = 'smtp.gmail.com', 
                   imap_server: str = 'imap.gmail.com', 
                   username: str = '', password: str = '') -> Dict[str, Any]:
    """
    Get comprehensive email statistics and account information.
    
    Args:
        smtp_server: SMTP server address
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with comprehensive email statistics
    """
    try:
        config = EmailConfig(
            smtp_server=smtp_server,
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        tool = EmailTool(config)
        return tool.get_comprehensive_stats()
    except Exception as e:
        return {'status': 'error', 'message': f'Failed to get email stats: {str(e)}'}

@register_tool(tags=["email", "agent", "advanced", "wrapper"])
def create_email_agent(smtp_server: str = 'smtp.gmail.com', 
                      imap_server: str = 'imap.gmail.com',
                      username: str = '', password: str = '') -> Dict[str, Any]:
    """
    Create an email agent for complex operations.
    
    Args:
        smtp_server: SMTP server address
        imap_server: IMAP server address
        username: Email username
        password: Email password
        
    Returns:
        Dictionary with agent information and capabilities
    """
    try:
        config = EmailConfig(
            smtp_server=smtp_server,
            imap_server=imap_server,
            username=username,
            password=password
        )
        
        agent = EmailAgent(config)
        
        return {
            'status': 'success',
            'agent_id': id(agent),
            'capabilities': agent.get_capabilities(),
            'config': {
                'smtp_server': smtp_server,
                'imap_server': imap_server,
                'username': username
            }
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Agent creation failed: {str(e)}'}