"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Cloud Tool - Manages cloud service interactions including storage operations, compute resources, and service deployments across major cloud providers.
"""

import os
import json
import time
import logging
import hashlib
import tempfile
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, IO, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import uuid
import re
import base64
from urllib.parse import urlparse
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import LLMFlow registration decorator
from llmflow.tools.tool_decorator import register_tool

# Optional imports with fallbacks
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload, MediaIoBaseUpload
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    from docx.shared import Inches
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

@dataclass
class GoogleConfig:
    
    credentials_file: str = os.getenv('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
    token_file: str = os.getenv('GOOGLE_TOKEN_FILE', 'token.json')
    scopes: List[str] = None
    timeout: int = 30
    chunk_size: int = 1024 * 1024  # 1MB chunks for uploads

@dataclass
class FileMetadata:
    """File metadata structure."""
    id: str
    name: str
    mime_type: str
    size: Optional[int]
    created_time: datetime
    modified_time: datetime
    parent_folders: List[str]
    shared: bool
    permissions: List[Dict[str, Any]]
    web_view_link: Optional[str]
    web_content_link: Optional[str]
    thumbnail_link: Optional[str]
    description: Optional[str]
    version: Optional[str]

@dataclass
class DocumentContent:
    """Document content structure."""
    document_id: str
    title: str
    body: Dict[str, Any]
    revision_id: str
    created_time: datetime
    modified_time: datetime
    authors: List[str]
    plain_text: str
    formatted_content: Dict[str, Any]

@dataclass
class SpreadsheetData:
    """Spreadsheet data structure."""
    spreadsheet_id: str
    title: str
    sheets: List[Dict[str, Any]]
    data: Dict[str, Any]  # sheet_name -> data
    properties: Dict[str, Any]
    created_time: datetime
    modified_time: datetime

class GoogleDriveService:
    """
    Google Drive service for file management.
    """
    
    def __init__(self, config: GoogleConfig):
        self.config = config
        self.service = None
        self.credentials = None
        self.file_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        if GOOGLE_API_AVAILABLE:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        try:
            # Default scopes if not provided
            if not self.config.scopes:
                self.config.scopes = [
                    'https://www.googleapis.com/auth/drive',
                    'https://www.googleapis.com/auth/documents',
                    'https://www.googleapis.com/auth/spreadsheets'
                ]
            
            creds = None
            
            # Load existing token
            if os.path.exists(self.config.token_file):
                creds = Credentials.from_authorized_user_file(
                    self.config.token_file, self.config.scopes
                )
            
            # If there are no valid credentials, get them
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.config.credentials_file):
                        raise FileNotFoundError(f"Credentials file not found: {self.config.credentials_file}")
                    
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.config.credentials_file, self.config.scopes
                    )
                    creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(self.config.token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.credentials = creds
            self.service = build('drive', 'v3', credentials=creds)
            
        except Exception as e:
            logging.error(f"Google Drive authentication failed: {e}")
            raise
    
    def test_connection(self) -> Dict[str, Any]:
        """Test Google Drive connection."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Google Drive service not initialized'}
            
            about = self.service.about().get(fields='user,storageQuota').execute()
            
            return {
                'status': 'success',
                'user': about['user']['displayName'],
                'email': about['user']['emailAddress'],
                'storage_quota': about['storageQuota'],
                'message': 'Google Drive connection successful'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Connection test failed: {str(e)}'}
    
    def upload_file(self, file_path: str, parent_folder_id: str = None,
                   file_name: str = None, description: str = None,
                   convert_to_google_format: bool = False) -> Dict[str, Any]:
        """
        Upload file to Google Drive.
        
        Args:
            file_path: Path to local file
            parent_folder_id: ID of parent folder (optional)
            file_name: Custom file name (optional)
            description: File description (optional)
            convert_to_google_format: Convert to Google format (Docs, Sheets, etc.)
        """
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            if not file_name:
                file_name = os.path.basename(file_path)
            
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # File metadata
            file_metadata = {
                'name': file_name,
                'description': description or ''
            }
            
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            # Convert to Google format if requested
            if convert_to_google_format:
                conversion_map = {
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'application/vnd.google-apps.document',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'application/vnd.google-apps.spreadsheet',
                    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'application/vnd.google-apps.presentation',
                    'text/csv': 'application/vnd.google-apps.spreadsheet',
                    'text/plain': 'application/vnd.google-apps.document'
                }
                
                if mime_type in conversion_map:
                    file_metadata['mimeType'] = conversion_map[mime_type]
            
            # Upload file
            media = MediaFileUpload(
                file_path,
                mimetype=mime_type,
                chunksize=self.config.chunk_size,
                resumable=True
            )
            
            request = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,mimeType,size,webViewLink,webContentLink'
            )
            
            file_obj = None
            while file_obj is None:
                status, file_obj = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logging.info(f"Upload progress: {progress}%")
            
            return {
                'status': 'success',
                'file_id': file_obj['id'],
                'file_name': file_obj['name'],
                'mime_type': file_obj['mimeType'],
                'size': file_obj.get('size'),
                'web_view_link': file_obj.get('webViewLink'),
                'web_content_link': file_obj.get('webContentLink'),
                'message': f'File "{file_name}" uploaded successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Upload failed: {str(e)}'}
    
    def download_file(self, file_id: str, output_path: str = None,
                     export_format: str = None) -> Dict[str, Any]:
        """
        Download file from Google Drive.
        
        Args:
            file_id: Google Drive file ID
            output_path: Local output path (optional)
            export_format: Export format for Google Docs/Sheets (optional)
        """
        try:
            # Get file metadata
            file_metadata = self.service.files().get(fileId=file_id).execute()
            
            if not output_path:
                output_path = file_metadata['name']
            
            # Handle Google Workspace files (Docs, Sheets, Slides)
            if file_metadata['mimeType'].startswith('application/vnd.google-apps.'):
                if not export_format:
                    # Default export formats
                    export_formats = {
                        'application/vnd.google-apps.document': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'application/vnd.google-apps.spreadsheet': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                        'application/vnd.google-apps.presentation': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
                    }
                    export_format = export_formats.get(file_metadata['mimeType'], 'application/pdf')
                
                request = self.service.files().export_media(
                    fileId=file_id,
                    mimeType=export_format
                )
            else:
                request = self.service.files().get_media(fileId=file_id)
            
            # Download file
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request, chunksize=self.config.chunk_size)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logging.info(f"Download progress: {progress}%")
            
            # Save to file
            with open(output_path, 'wb') as f:
                f.write(fh.getvalue())
            
            file_size = os.path.getsize(output_path)
            
            return {
                'status': 'success',
                'file_id': file_id,
                'file_name': file_metadata['name'],
                'output_path': output_path,
                'file_size': file_size,
                'export_format': export_format,
                'message': f'File downloaded to {output_path}'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Download failed: {str(e)}'}
    
    def create_folder(self, folder_name: str, parent_folder_id: str = None,
                     description: str = None) -> Dict[str, Any]:
        """Create folder in Google Drive."""
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'description': description or ''
            }
            
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            folder = self.service.files().create(
                body=file_metadata,
                fields='id,name,webViewLink'
            ).execute()
            
            return {
                'status': 'success',
                'folder_id': folder['id'],
                'folder_name': folder['name'],
                'web_view_link': folder.get('webViewLink'),
                'message': f'Folder "{folder_name}" created successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create folder: {str(e)}'}
    
    def list_files(self, folder_id: str = None, query: str = None,
                  page_size: int = 100, order_by: str = 'modifiedTime desc') -> Dict[str, Any]:
        """
        List files in Google Drive.
        
        Args:
            folder_id: Parent folder ID (optional, defaults to root)
            query: Search query (optional)
            page_size: Number of files per page
            order_by: Sort order
        """
        try:
            # Build query
            query_parts = []
            
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            
            if query:
                query_parts.append(f"name contains '{query}'")
            
            # Add default filters
            query_parts.append("trashed=false")
            
            full_query = ' and '.join(query_parts)
            
            # List files
            results = self.service.files().list(
                q=full_query,
                pageSize=page_size,
                orderBy=order_by,
                fields='nextPageToken,files(id,name,mimeType,size,createdTime,modifiedTime,parents,shared,webViewLink,thumbnailLink,description)'
            ).execute()
            
            files = results.get('files', [])
            
            # Convert to FileMetadata objects
            file_list = []
            for file_data in files:
                file_metadata = FileMetadata(
                    id=file_data['id'],
                    name=file_data['name'],
                    mime_type=file_data['mimeType'],
                    size=int(file_data.get('size', 0)) if file_data.get('size') else None,
                    created_time=datetime.fromisoformat(file_data['createdTime'].replace('Z', '+00:00')),
                    modified_time=datetime.fromisoformat(file_data['modifiedTime'].replace('Z', '+00:00')),
                    parent_folders=file_data.get('parents', []),
                    shared=file_data.get('shared', False),
                    permissions=[],  # Would need separate API call
                    web_view_link=file_data.get('webViewLink'),
                    web_content_link=file_data.get('webContentLink'),
                    thumbnail_link=file_data.get('thumbnailLink'),
                    description=file_data.get('description'),
                    version=None
                )
                file_list.append(asdict(file_metadata))
            
            return {
                'status': 'success',
                'files': file_list,
                'total_files': len(file_list),
                'next_page_token': results.get('nextPageToken'),
                'query': full_query
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to list files: {str(e)}'}
    
    def search_files(self, query: str, file_type: str = None,
                    modified_after: datetime = None) -> Dict[str, Any]:
        """
        Search files in Google Drive.
        
        Args:
            query: Search query
            file_type: MIME type filter (optional)
            modified_after: Only files modified after this date (optional)
        """
        try:
            query_parts = [f"name contains '{query}'"]
            
            if file_type:
                query_parts.append(f"mimeType='{file_type}'")
            
            if modified_after:
                date_str = modified_after.isoformat() + 'Z'
                query_parts.append(f"modifiedTime > '{date_str}'")
            
            query_parts.append("trashed=false")
            
            search_query = ' and '.join(query_parts)
            
            return self.list_files(query=search_query)
            
        except Exception as e:
            return {'status': 'error', 'message': f'Search failed: {str(e)}'}
    
    def delete_file(self, file_id: str, permanent: bool = False) -> Dict[str, Any]:
        """
        Delete file from Google Drive.
        
        Args:
            file_id: File ID to delete
            permanent: If True, delete permanently; if False, move to trash
        """
        try:
            if permanent:
                self.service.files().delete(fileId=file_id).execute()
                message = 'File deleted permanently'
            else:
                self.service.files().update(
                    fileId=file_id,
                    body={'trashed': True}
                ).execute()
                message = 'File moved to trash'
            
            return {
                'status': 'success',
                'file_id': file_id,
                'permanent': permanent,
                'message': message
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete file: {str(e)}'}
    
    def share_file(self, file_id: str, email: str = None, role: str = 'reader',
                  link_sharing: bool = False) -> Dict[str, Any]:
        """
        Share file with user or make it publicly accessible.
        
        Args:
            file_id: File ID to share
            email: Email address to share with (optional)
            role: Permission role (reader, writer, owner)
            link_sharing: Make accessible to anyone with link
        """
        try:
            permissions = []
            
            if email:
                permission = {
                    'type': 'user',
                    'role': role,
                    'emailAddress': email
                }
                
                result = self.service.permissions().create(
                    fileId=file_id,
                    body=permission,
                    sendNotificationEmail=True
                ).execute()
                
                permissions.append(result)
            
            if link_sharing:
                permission = {
                    'type': 'anyone',
                    'role': role
                }
                
                result = self.service.permissions().create(
                    fileId=file_id,
                    body=permission
                ).execute()
                
                permissions.append(result)
            
            # Get updated file info
            file_info = self.service.files().get(
                fileId=file_id,
                fields='webViewLink,webContentLink'
            ).execute()
            
            return {
                'status': 'success',
                'file_id': file_id,
                'permissions': permissions,
                'web_view_link': file_info.get('webViewLink'),
                'web_content_link': file_info.get('webContentLink'),
                'message': 'File shared successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to share file: {str(e)}'}
    
    def copy_file(self, file_id: str, new_name: str, parent_folder_id: str = None) -> Dict[str, Any]:
        """Copy file in Google Drive."""
        try:
            file_metadata = {'name': new_name}
            
            if parent_folder_id:
                file_metadata['parents'] = [parent_folder_id]
            
            copied_file = self.service.files().copy(
                fileId=file_id,
                body=file_metadata,
                fields='id,name,webViewLink'
            ).execute()
            
            return {
                'status': 'success',
                'original_file_id': file_id,
                'new_file_id': copied_file['id'],
                'new_file_name': copied_file['name'],
                'web_view_link': copied_file.get('webViewLink'),
                'message': f'File copied as "{new_name}"'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to copy file: {str(e)}'}

class GoogleDocsService:
    """
    Google Docs service for document management.
    """
    
    def __init__(self, config: GoogleConfig, drive_service: GoogleDriveService):
        self.config = config
        self.drive_service = drive_service
        self.service = None
        
        if GOOGLE_API_AVAILABLE and drive_service.credentials:
            self.service = build('docs', 'v1', credentials=drive_service.credentials)
    
    def create_document(self, title: str, content: str = None,
                       parent_folder_id: str = None) -> Dict[str, Any]:
        """Create new Google Document."""
        try:
            # Create document
            document = self.service.documents().create(
                body={'title': title}
            ).execute()
            
            document_id = document['documentId']
            
            # Add content if provided
            if content:
                self.insert_text(document_id, content, index=1)
            
            # Move to specified folder if provided
            if parent_folder_id:
                self.drive_service.service.files().update(
                    fileId=document_id,
                    addParents=parent_folder_id,
                    fields='id,parents'
                ).execute()
            
            # Get document info
            doc_info = self.drive_service.service.files().get(
                fileId=document_id,
                fields='webViewLink,createdTime,modifiedTime'
            ).execute()
            
            return {
                'status': 'success',
                'document_id': document_id,
                'title': title,
                'web_view_link': doc_info.get('webViewLink'),
                'created_time': doc_info.get('createdTime'),
                'message': f'Document "{title}" created successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create document: {str(e)}'}
    
    def get_document(self, document_id: str, include_content: bool = True) -> Dict[str, Any]:
        """Get Google Document content and metadata."""
        try:
            # Get document
            document = self.service.documents().get(documentId=document_id).execute()
            
            # Extract text content
            plain_text = ''
            if include_content and 'body' in document:
                plain_text = self._extract_text_from_document(document)
            
            # Get file metadata from Drive
            file_info = self.drive_service.service.files().get(
                fileId=document_id,
                fields='createdTime,modifiedTime,owners,webViewLink'
            ).execute()
            
            # Extract authors
            authors = []
            if 'owners' in file_info:
                authors = [owner['displayName'] for owner in file_info['owners']]
            
            doc_content = DocumentContent(
                document_id=document_id,
                title=document['title'],
                body=document.get('body', {}) if include_content else {},
                revision_id=document['revisionId'],
                created_time=datetime.fromisoformat(file_info['createdTime'].replace('Z', '+00:00')),
                modified_time=datetime.fromisoformat(file_info['modifiedTime'].replace('Z', '+00:00')),
                authors=authors,
                plain_text=plain_text,
                formatted_content=document if include_content else {}
            )
            
            return {
                'status': 'success',
                'document': asdict(doc_content),
                'web_view_link': file_info.get('webViewLink')
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get document: {str(e)}'}
    
    def _extract_text_from_document(self, document: Dict[str, Any]) -> str:
        """Extract plain text from document structure."""
        text_content = []
        
        def extract_text_from_elements(elements):
            for element in elements:
                if 'paragraph' in element:
                    paragraph = element['paragraph']
                    if 'elements' in paragraph:
                        for para_element in paragraph['elements']:
                            if 'textRun' in para_element:
                                text_content.append(para_element['textRun']['content'])
                elif 'table' in element:
                    table = element['table']
                    for row in table.get('tableRows', []):
                        for cell in row.get('tableCells', []):
                            if 'content' in cell:
                                extract_text_from_elements(cell['content'])
        
        if 'body' in document and 'content' in document['body']:
            extract_text_from_elements(document['body']['content'])
        
        return ''.join(text_content)
    
    def insert_text(self, document_id: str, text: str, index: int = 1) -> Dict[str, Any]:
        """Insert text into document at specified index."""
        try:
            requests = [{
                'insertText': {
                    'location': {'index': index},
                    'text': text
                }
            }]
            
            result = self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return {
                'status': 'success',
                'document_id': document_id,
                'text_length': len(text),
                'insert_index': index,
                'message': 'Text inserted successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to insert text: {str(e)}'}
    
    def replace_text(self, document_id: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Replace text in document."""
        try:
            requests = [{
                'replaceAllText': {
                    'containsText': {
                        'text': old_text,
                        'matchCase': False
                    },
                    'replaceText': new_text
                }
            }]
            
            result = self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            replacements = result.get('replies', [{}])[0].get('replaceAllText', {}).get('occurrencesChanged', 0)
            
            return {
                'status': 'success',
                'document_id': document_id,
                'old_text': old_text,
                'new_text': new_text,
                'replacements_made': replacements,
                'message': f'Replaced {replacements} occurrences'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to replace text: {str(e)}'}
    
    def format_text(self, document_id: str, start_index: int, end_index: int,
                   bold: bool = None, italic: bool = None, font_size: int = None,
                   font_family: str = None) -> Dict[str, Any]:
        """Format text in document."""
        try:
            text_style = {}
            
            if bold is not None:
                text_style['bold'] = bold
            if italic is not None:
                text_style['italic'] = italic
            if font_size is not None:
                text_style['fontSize'] = {'magnitude': font_size, 'unit': 'PT'}
            if font_family is not None:
                text_style['weightedFontFamily'] = {'fontFamily': font_family}
            
            requests = [{
                'updateTextStyle': {
                    'range': {
                        'startIndex': start_index,
                        'endIndex': end_index
                    },
                    'textStyle': text_style,
                    'fields': ','.join(text_style.keys())
                }
            }]
            
            result = self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return {
                'status': 'success',
                'document_id': document_id,
                'start_index': start_index,
                'end_index': end_index,
                'formatting_applied': text_style,
                'message': 'Text formatting applied successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to format text: {str(e)}'}
    
    def insert_image(self, document_id: str, image_url: str, index: int,
                    width: float = None, height: float = None) -> Dict[str, Any]:
        """Insert image into document."""
        try:
            requests = [{
                'insertInlineImage': {
                    'location': {'index': index},
                    'uri': image_url
                }
            }]
            
            # Add size if specified
            if width or height:
                requests.append({
                    'updateInlineImageSize': {
                        'location': {'index': index},
                        'size': {
                            'width': {'magnitude': width, 'unit': 'PT'} if width else None,
                            'height': {'magnitude': height, 'unit': 'PT'} if height else None
                        }
                    }
                })
            
            result = self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return {
                'status': 'success',
                'document_id': document_id,
                'image_url': image_url,
                'insert_index': index,
                'message': 'Image inserted successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to insert image: {str(e)}'}
    
    def create_table(self, document_id: str, rows: int, columns: int,
                    index: int) -> Dict[str, Any]:
        """Create table in document."""
        try:
            requests = [{
                'insertTable': {
                    'location': {'index': index},
                    'rows': rows,
                    'columns': columns
                }
            }]
            
            result = self.service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return {
                'status': 'success',
                'document_id': document_id,
                'rows': rows,
                'columns': columns,
                'insert_index': index,
                'message': f'Table ({rows}x{columns}) created successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create table: {str(e)}'}

class GoogleSheetsService:
    """
    Google Sheets service for spreadsheet management.
    """
    
    def __init__(self, config: GoogleConfig, drive_service: GoogleDriveService):
        self.config = config
        self.drive_service = drive_service
        self.service = None
        
        if GOOGLE_API_AVAILABLE and drive_service.credentials:
            self.service = build('sheets', 'v4', credentials=drive_service.credentials)
    
    def create_spreadsheet(self, title: str, sheet_names: List[str] = None,
                          parent_folder_id: str = None) -> Dict[str, Any]:
        """Create new Google Spreadsheet."""
        try:
            # Prepare sheets
            sheets = []
            if sheet_names:
                for i, sheet_name in enumerate(sheet_names):
                    sheets.append({
                        'properties': {
                            'sheetId': i,
                            'title': sheet_name
                        }
                    })
            else:
                sheets.append({
                    'properties': {
                        'sheetId': 0,
                        'title': 'Sheet1'
                    }
                })
            
            # Create spreadsheet
            spreadsheet_body = {
                'properties': {'title': title},
                'sheets': sheets
            }
            
            spreadsheet = self.service.spreadsheets().create(
                body=spreadsheet_body
            ).execute()
            
            spreadsheet_id = spreadsheet['spreadsheetId']
            
            # Move to specified folder if provided
            if parent_folder_id:
                self.drive_service.service.files().update(
                    fileId=spreadsheet_id,
                    addParents=parent_folder_id,
                    fields='id,parents'
                ).execute()
            
            # Get spreadsheet info
            sheet_info = self.drive_service.service.files().get(
                fileId=spreadsheet_id,
                fields='webViewLink,createdTime,modifiedTime'
            ).execute()
            
            return {
                'status': 'success',
                'spreadsheet_id': spreadsheet_id,
                'title': title,
                'sheets': [sheet['properties']['title'] for sheet in sheets],
                'web_view_link': sheet_info.get('webViewLink'),
                'created_time': sheet_info.get('createdTime'),
                'message': f'Spreadsheet "{title}" created successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create spreadsheet: {str(e)}'}
    
    def get_spreadsheet(self, spreadsheet_id: str, include_data: bool = True) -> Dict[str, Any]:
        """Get spreadsheet metadata and data."""
        try:
            # Get spreadsheet metadata
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=spreadsheet_id,
                includeGridData=include_data
            ).execute()
            
            # Get file metadata from Drive
            file_info = self.drive_service.service.files().get(
                fileId=spreadsheet_id,
                fields='createdTime,modifiedTime,webViewLink'
            ).execute()
            
            # Extract sheet data
            sheet_data = {}
            if include_data:
                for sheet in spreadsheet.get('sheets', []):
                    sheet_name = sheet['properties']['title']
                    if 'data' in sheet and sheet['data']:
                        sheet_data[sheet_name] = self._extract_sheet_data(sheet['data'][0])
            
            spreadsheet_data = SpreadsheetData(
                spreadsheet_id=spreadsheet_id,
                title=spreadsheet['properties']['title'],
                sheets=[sheet['properties'] for sheet in spreadsheet.get('sheets', [])],
                data=sheet_data,
                properties=spreadsheet['properties'],
                created_time=datetime.fromisoformat(file_info['createdTime'].replace('Z', '+00:00')),
                modified_time=datetime.fromisoformat(file_info['modifiedTime'].replace('Z', '+00:00'))
            )
            
            return {
                'status': 'success',
                'spreadsheet': asdict(spreadsheet_data),
                'web_view_link': file_info.get('webViewLink')
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get spreadsheet: {str(e)}'}
    
    def _extract_sheet_data(self, grid_data: Dict[str, Any]) -> List[List[Any]]:
        """Extract data from sheet grid data."""
        rows = []
        
        for row_data in grid_data.get('rowData', []):
            row = []
            for cell_data in row_data.get('values', []):
                if 'effectiveValue' in cell_data:
                    value = list(cell_data['effectiveValue'].values())[0]
                    row.append(value)
                else:
                    row.append('')
            rows.append(row)
        
        return rows
    
    def read_range(self, spreadsheet_id: str, range_name: str,
                  value_render_option: str = 'UNFORMATTED_VALUE') -> Dict[str, Any]:
        """Read data from specific range."""
        try:
            result = self.service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueRenderOption=value_render_option
            ).execute()
            
            values = result.get('values', [])
            
            return {
                'status': 'success',
                'spreadsheet_id': spreadsheet_id,
                'range': range_name,
                'values': values,
                'total_rows': len(values),
                'total_columns': len(values[0]) if values else 0
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to read range: {str(e)}'}
    
    def write_range(self, spreadsheet_id: str, range_name: str,
                   values: List[List[Any]], value_input_option: str = 'RAW') -> Dict[str, Any]:
        """Write data to specific range."""
        try:
            body = {
                'values': values
            }
            
            result = self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                body=body
            ).execute()
            
            return {
                'status': 'success',
                'spreadsheet_id': spreadsheet_id,
                'range': range_name,
                'updated_rows': result.get('updatedRows', 0),
                'updated_columns': result.get('updatedColumns', 0),
                'updated_cells': result.get('updatedCells', 0),
                'message': f'Updated {result.get("updatedCells", 0)} cells'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to write range: {str(e)}'}
    
    def append_data(self, spreadsheet_id: str, range_name: str,
                   values: List[List[Any]], value_input_option: str = 'RAW') -> Dict[str, Any]:
        """Append data to spreadsheet."""
        try:
            body = {
                'values': values
            }
            
            result = self.service.spreadsheets().values().append(
                spreadsheetId=spreadsheet_id,
                range=range_name,
                valueInputOption=value_input_option,
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            return {
                'status': 'success',
                'spreadsheet_id': spreadsheet_id,
                'range': range_name,
                'updated_rows': result.get('updates', {}).get('updatedRows', 0),
                'updated_cells': result.get('updates', {}).get('updatedCells', 0),
                'message': f'Appended {len(values)} rows'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to append data: {str(e)}'}
    
    def create_chart(self, spreadsheet_id: str, sheet_id: int, chart_type: str,
                    data_range: str, title: str = None) -> Dict[str, Any]:
        """Create chart in spreadsheet."""
        try:
            # Parse range
            range_parts = data_range.split(':')
            start_cell = range_parts[0]
            end_cell = range_parts[1] if len(range_parts) > 1 else start_cell
            
            # Convert A1 notation to row/column indices
            def a1_to_indices(cell):
                col_str = re.match(r'[A-Z]+', cell).group()
                row_str = re.match(r'[A-Z]+(\d+)', cell).group(1)
                
                col_num = 0
                for char in col_str:
                    col_num = col_num * 26 + (ord(char) - ord('A') + 1)
                
                return int(row_str) - 1, col_num - 1
            
            start_row, start_col = a1_to_indices(start_cell)
            end_row, end_col = a1_to_indices(end_cell)
            
            requests = [{
                'addChart': {
                    'chart': {
                        'spec': {
                            'chartType': chart_type.upper(),
                            'title': title or 'Chart',
                            'basicChart': {
                                'chartType': chart_type.upper(),
                                'domains': [{
                                    'domain': {
                                        'sourceRange': {
                                            'sources': [{
                                                'sheetId': sheet_id,
                                                'startRowIndex': start_row,
                                                'endRowIndex': end_row + 1,
                                                'startColumnIndex': start_col,
                                                'endColumnIndex': start_col + 1
                                            }]
                                        }
                                    }
                                }],
                                'series': [{
                                    'series': {
                                        'sourceRange': {
                                            'sources': [{
                                                'sheetId': sheet_id,
                                                'startRowIndex': start_row,
                                                'endRowIndex': end_row + 1,
                                                'startColumnIndex': start_col + 1,
                                                'endColumnIndex': end_col + 1
                                            }]
                                        }
                                    }
                                }]
                            }
                        },
                        'position': {
                            'overlayPosition': {
                                'anchorCell': {
                                    'sheetId': sheet_id,
                                    'rowIndex': end_row + 2,
                                    'columnIndex': start_col
                                }
                            }
                        }
                    }
                }
            }]
            
            result = self.service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={'requests': requests}
            ).execute()
            
            chart_id = result['replies'][0]['addChart']['chart']['chartId']
            
            return {
                'status': 'success',
                'spreadsheet_id': spreadsheet_id,
                'chart_id': chart_id,
                'chart_type': chart_type,
                'data_range': data_range,
                'message': f'Chart created successfully'
            }
            
        except HttpError as e:
            return {'status': 'error', 'message': f'Google API error: {e}'}
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create chart: {str(e)}'}

class GoogleCloudServicesTool:
    """
    Comprehensive Google Cloud services tool.
    """
    
    def __init__(self):
        self.config = None
        self.drive_service = None
        self.docs_service = None
        self.sheets_service = None
        
        # Operation history
        self.operation_history = []
        
        # Statistics
        self.stats = {
            'files_uploaded': 0,
            'files_downloaded': 0,
            'documents_created': 0,
            'spreadsheets_created': 0,
            'total_storage_used': 0,
            'last_activity': None
        }
        
        # Cache
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def configure(self, credentials_file: str, token_file: str = 'token.json',
                 scopes: List[str] = None) -> Dict[str, Any]:
        """Configure Google Cloud services."""
        try:
            self.config = GoogleConfig(
                credentials_file=credentials_file,
                token_file=token_file,
                scopes=scopes
            )
            
            # Initialize services
            self.drive_service = GoogleDriveService(self.config)
            self.docs_service = GoogleDocsService(self.config, self.drive_service)
            self.sheets_service = GoogleSheetsService(self.config, self.drive_service)
            
            # Test connection
            test_result = self.drive_service.test_connection()
            
            if test_result['status'] == 'success':
                return {
                    'status': 'success',
                    'message': 'Google Cloud services configured successfully',
                    'user': test_result.get('user'),
                    'email': test_result.get('email'),
                    'services': ['drive', 'docs', 'sheets']
                }
            else:
                return test_result
                
        except Exception as e:
            return {'status': 'error', 'message': f'Configuration failed: {str(e)}'}
    
    def upload_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """Upload file to Google Drive."""
        try:
            if not self.drive_service:
                return {'status': 'error', 'message': 'Drive service not configured'}
            
            result = self.drive_service.upload_file(file_path, **kwargs)
            
            if result['status'] == 'success':
                self.stats['files_uploaded'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
                
                if result.get('size'):
                    self.stats['total_storage_used'] += int(result['size'])
                
                # Record operation
                self.operation_history.append({
                    'type': 'upload_file',
                    'file_path': file_path,
                    'file_id': result['file_id'],
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Upload failed: {str(e)}'}
    
    def download_file(self, file_id: str, **kwargs) -> Dict[str, Any]:
        """Download file from Google Drive."""
        try:
            if not self.drive_service:
                return {'status': 'error', 'message': 'Drive service not configured'}
            
            result = self.drive_service.download_file(file_id, **kwargs)
            
            if result['status'] == 'success':
                self.stats['files_downloaded'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Download failed: {str(e)}'}
    
    def create_document(self, title: str, content: str = None, **kwargs) -> Dict[str, Any]:
        """Create Google Document."""
        try:
            if not self.docs_service:
                return {'status': 'error', 'message': 'Docs service not configured'}
            
            result = self.docs_service.create_document(title, content, **kwargs)
            
            if result['status'] == 'success':
                self.stats['documents_created'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Document creation failed: {str(e)}'}
    
    def create_spreadsheet(self, title: str, **kwargs) -> Dict[str, Any]:
        """Create Google Spreadsheet."""
        try:
            if not self.sheets_service:
                return {'status': 'error', 'message': 'Sheets service not configured'}
            
            result = self.sheets_service.create_spreadsheet(title, **kwargs)
            
            if result['status'] == 'success':
                self.stats['spreadsheets_created'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Spreadsheet creation failed: {str(e)}'}
    
    def search_files(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search files in Google Drive."""
        try:
            if not self.drive_service:
                return {'status': 'error', 'message': 'Drive service not configured'}
            
            return self.drive_service.search_files(query, **kwargs)
            
        except Exception as e:
            return {'status': 'error', 'message': f'Search failed: {str(e)}'}
    
    def batch_upload(self, file_paths: List[str], parent_folder_id: str = None,
                    progress_callback: Callable = None) -> Dict[str, Any]:
        """Upload multiple files in batch."""
        try:
            results = []
            successful_uploads = 0
            failed_uploads = 0
            
            for i, file_path in enumerate(file_paths):
                try:
                    result = self.upload_file(file_path, parent_folder_id=parent_folder_id)
                    
                    results.append({
                        'file_path': file_path,
                        'status': result['status'],
                        'file_id': result.get('file_id'),
                        'message': result.get('message'),
                        'error': result.get('message') if result['status'] == 'error' else None
                    })
                    
                    if result['status'] == 'success':
                        successful_uploads += 1
                    else:
                        failed_uploads += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress = (i + 1) / len(file_paths) * 100
                        progress_callback(progress, file_path, result)
                        
                except Exception as e:
                    results.append({
                        'file_path': file_path,
                        'status': 'error',
                        'error': str(e)
                    })
                    failed_uploads += 1
            
            return {
                'status': 'success',
                'total_files': len(file_paths),
                'successful_uploads': successful_uploads,
                'failed_uploads': failed_uploads,
                'success_rate': (successful_uploads / len(file_paths)) * 100,
                'results': results
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Batch upload failed: {str(e)}'}
    
    def sync_folder(self, local_folder: str, drive_folder_id: str = None,
                   bidirectional: bool = False) -> Dict[str, Any]:
        """Sync local folder with Google Drive folder."""
        try:
            if not os.path.exists(local_folder):
                return {'status': 'error', 'message': f'Local folder not found: {local_folder}'}
            
            # Create drive folder if not exists
            if not drive_folder_id:
                folder_name = os.path.basename(local_folder)
                folder_result = self.drive_service.create_folder(folder_name)
                if folder_result['status'] != 'success':
                    return folder_result
                drive_folder_id = folder_result['folder_id']
            
            # Get local files
            local_files = []
            for root, dirs, files in os.walk(local_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, local_folder)
                    local_files.append({
                        'path': file_path,
                        'name': file,
                        'relative_path': rel_path,
                        'modified_time': datetime.fromtimestamp(os.path.getmtime(file_path))
                    })
            
            # Get drive files
            drive_files_result = self.drive_service.list_files(folder_id=drive_folder_id)
            if drive_files_result['status'] != 'success':
                return drive_files_result
            
            drive_files = {f['name']: f for f in drive_files_result['files']}
            
            # Upload new/modified files
            uploaded_files = []
            for local_file in local_files:
                file_name = local_file['name']
                
                should_upload = False
                if file_name not in drive_files:
                    should_upload = True
                else:
                    drive_modified = datetime.fromisoformat(
                        drive_files[file_name]['modified_time'].replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                    if local_file['modified_time'] > drive_modified:
                        should_upload = True
                
                if should_upload:
                    upload_result = self.upload_file(
                        local_file['path'],
                        parent_folder_id=drive_folder_id,
                        file_name=file_name
                    )
                    uploaded_files.append({
                        'file_name': file_name,
                        'result': upload_result
                    })
            
            # Download new/modified files (if bidirectional)
            downloaded_files = []
            if bidirectional:
                for drive_file_name, drive_file in drive_files.items():
                    local_file_path = os.path.join(local_folder, drive_file_name)
                    
                    should_download = False
                    if not os.path.exists(local_file_path):
                        should_download = True
                    else:
                        local_modified = datetime.fromtimestamp(os.path.getmtime(local_file_path))
                        drive_modified = datetime.fromisoformat(
                            drive_file['modified_time'].replace('Z', '+00:00')
                        ).replace(tzinfo=None)
                        if drive_modified > local_modified:
                            should_download = True
                    
                    if should_download:
                        download_result = self.download_file(
                            drive_file['id'],
                            output_path=local_file_path
                        )
                        downloaded_files.append({
                            'file_name': drive_file_name,
                            'result': download_result
                        })
            
            return {
                'status': 'success',
                'local_folder': local_folder,
                'drive_folder_id': drive_folder_id,
                'uploaded_files': len(uploaded_files),
                'downloaded_files': len(downloaded_files),
                'upload_details': uploaded_files,
                'download_details': downloaded_files,
                'message': f'Sync completed: {len(uploaded_files)} uploaded, {len(downloaded_files)} downloaded'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Folder sync failed: {str(e)}'}
    
    def export_to_pdf(self, file_id: str, output_path: str = None) -> Dict[str, Any]:
        """Export Google Docs/Sheets to PDF."""
        try:
            return self.download_file(
                file_id,
                output_path=output_path,
                export_format='application/pdf'
            )
            
        except Exception as e:
            return {'status': 'error', 'message': f'PDF export failed: {str(e)}'}
    
    def backup_drive(self, backup_folder: str, file_types: List[str] = None) -> Dict[str, Any]:
        """Backup Google Drive files to local folder."""
        try:
            os.makedirs(backup_folder, exist_ok=True)
            
            # Get all files
            all_files_result = self.drive_service.list_files(page_size=1000)
            if all_files_result['status'] != 'success':
                return all_files_result
            
            files_to_backup = all_files_result['files']
            
            # Filter by file types if specified
            if file_types:
                files_to_backup = [
                    f for f in files_to_backup 
                    if f['mime_type'] in file_types
                ]
            
            # Download files
            backup_results = []
            successful_backups = 0
            
            for file_info in files_to_backup:
                try:
                    safe_filename = re.sub(r'[<>:"/\\|?*]', '_', file_info['name'])
                    output_path = os.path.join(backup_folder, safe_filename)
                    
                    download_result = self.download_file(
                        file_info['id'],
                        output_path=output_path
                    )
                    
                    backup_results.append({
                        'file_name': file_info['name'],
                        'file_id': file_info['id'],
                        'status': download_result['status'],
                        'output_path': output_path if download_result['status'] == 'success' else None
                    })
                    
                    if download_result['status'] == 'success':
                        successful_backups += 1
                        
                except Exception as e:
                    backup_results.append({
                        'file_name': file_info['name'],
                        'file_id': file_info['id'],
                        'status': 'error',
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'backup_folder': backup_folder,
                'total_files': len(files_to_backup),
                'successful_backups': successful_backups,
                'failed_backups': len(files_to_backup) - successful_backups,
                'backup_results': backup_results
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Backup failed: {str(e)}'}
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get Google Drive storage usage statistics."""
        try:
            if not self.drive_service:
                return {'status': 'error', 'message': 'Drive service not configured'}
            
            about = self.drive_service.service.about().get(
                fields='storageQuota,user'
            ).execute()
            
            quota = about.get('storageQuota', {})
            
            total_bytes = int(quota.get('limit', 0))
            used_bytes = int(quota.get('usage', 0))
            available_bytes = total_bytes - used_bytes
            
            # Convert to human readable
            def bytes_to_human(bytes_value):
                for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                    if bytes_value < 1024.0:
                        return f"{bytes_value:.2f} {unit}"
                    bytes_value /= 1024.0
                return f"{bytes_value:.2f} PB"
            
            return {
                'status': 'success',
                'storage_quota': {
                    'total_bytes': total_bytes,
                    'used_bytes': used_bytes,
                    'available_bytes': available_bytes,
                    'total_human': bytes_to_human(total_bytes),
                    'used_human': bytes_to_human(used_bytes),
                    'available_human': bytes_to_human(available_bytes),
                    'usage_percentage': (used_bytes / total_bytes * 100) if total_bytes > 0 else 0
                },
                'user': about.get('user', {})
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get storage usage: {str(e)}'}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'status': 'success',
            'stats': self.stats,
            'operation_history_count': len(self.operation_history),
            'cache_size': len(self.cache),
            'services_configured': {
                'drive': self.drive_service is not None,
                'docs': self.docs_service is not None,
                'sheets': self.sheets_service is not None
            },
            'available_libraries': {
                'google_api': GOOGLE_API_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'docx': DOCX_AVAILABLE,
                'pdf': PDF_AVAILABLE
            }
        }

# Agent framework integration
class GoogleCloudAgent:
    """
    Agent wrapper for Google Cloud services.
    """
    
    def __init__(self):
        self.tool = GoogleCloudServicesTool()
        self.capabilities = [
            'configure',
            'upload_file',
            'download_file',
            'create_document',
            'create_spreadsheet',
            'search_files',
            'list_files',
            'delete_file',
            'share_file',
            'create_folder',
            'batch_upload',
            'sync_folder',
            'export_pdf',
            'backup_drive',
            'get_storage_usage',
            'get_stats'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific Google Cloud operation."""
        try:
            if action == 'configure':
                return self.tool.configure(**kwargs)
            elif action == 'upload_file':
                return self.tool.upload_file(**kwargs)
            elif action == 'download_file':
                return self.tool.download_file(**kwargs)
            elif action == 'create_document':
                return self.tool.create_document(**kwargs)
            elif action == 'create_spreadsheet':
                return self.tool.create_spreadsheet(**kwargs)
            elif action == 'search_files':
                return self.tool.search_files(**kwargs)
            elif action == 'list_files':
                return self.tool.drive_service.list_files(**kwargs) if self.tool.drive_service else {'status': 'error', 'message': 'Drive service not configured'}
            elif action == 'delete_file':
                return self.tool.drive_service.delete_file(**kwargs) if self.tool.drive_service else {'status': 'error', 'message': 'Drive service not configured'}
            elif action == 'share_file':
                return self.tool.drive_service.share_file(**kwargs) if self.tool.drive_service else {'status': 'error', 'message': 'Drive service not configured'}
            elif action == 'create_folder':
                return self.tool.drive_service.create_folder(**kwargs) if self.tool.drive_service else {'status': 'error', 'message': 'Drive service not configured'}
            elif action == 'batch_upload':
                return self.tool.batch_upload(**kwargs)
            elif action == 'sync_folder':
                return self.tool.sync_folder(**kwargs)
            elif action == 'export_pdf':
                return self.tool.export_to_pdf(**kwargs)
            elif action == 'backup_drive':
                return self.tool.backup_drive(**kwargs)
            elif action == 'get_storage_usage':
                return self.tool.get_storage_usage()
            elif action == 'get_stats':
                return self.tool.get_comprehensive_stats()
            
            # Document operations
            elif action == 'get_document':
                return self.tool.docs_service.get_document(**kwargs) if self.tool.docs_service else {'status': 'error', 'message': 'Docs service not configured'}
            elif action == 'insert_text':
                return self.tool.docs_service.insert_text(**kwargs) if self.tool.docs_service else {'status': 'error', 'message': 'Docs service not configured'}
            elif action == 'replace_text':
                return self.tool.docs_service.replace_text(**kwargs) if self.tool.docs_service else {'status': 'error', 'message': 'Docs service not configured'}
            elif action == 'format_text':
                return self.tool.docs_service.format_text(**kwargs) if self.tool.docs_service else {'status': 'error', 'message': 'Docs service not configured'}
            
            # Spreadsheet operations
            elif action == 'get_spreadsheet':
                return self.tool.sheets_service.get_spreadsheet(**kwargs) if self.tool.sheets_service else {'status': 'error', 'message': 'Sheets service not configured'}
            elif action == 'read_range':
                return self.tool.sheets_service.read_range(**kwargs) if self.tool.sheets_service else {'status': 'error', 'message': 'Sheets service not configured'}
            elif action == 'write_range':
                return self.tool.sheets_service.write_range(**kwargs) if self.tool.sheets_service else {'status': 'error', 'message': 'Sheets service not configured'}
            elif action == 'append_data':
                return self.tool.sheets_service.append_data(**kwargs) if self.tool.sheets_service else {'status': 'error', 'message': 'Sheets service not configured'}
            
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()

# Quick utility functions
@register_tool(tags=["cloud", "google", "setup", "quick"])
def quick_setup(credentials_file: str, token_file: str = 'token.json') -> GoogleCloudAgent:
    """Quick setup of Google Cloud agent."""
    agent = GoogleCloudAgent()
    result = agent.execute('configure', 
                          credentials_file=credentials_file,
                          token_file=token_file)
    
    if result['status'] != 'success':
        raise Exception(f"Setup failed: {result['message']}")
    
    return agent

@register_tool(tags=["cloud", "google", "upload", "batch", "folder"])
def batch_upload_folder(agent: GoogleCloudAgent, folder_path: str, 
                       drive_folder_id: str = None) -> Dict[str, Any]:
    """Upload entire folder to Google Drive."""
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    return agent.execute('batch_upload', 
                        file_paths=file_paths,
                        parent_folder_id=drive_folder_id)

@register_tool(tags=["cloud", "google", "docs", "create", "content"])
def create_document_with_content(agent: GoogleCloudAgent, title: str, 
                               paragraphs: List[str]) -> Dict[str, Any]:
    """Create document with structured content."""
    content = '\n\n'.join(paragraphs)
    return agent.execute('create_document', title=title, content=content)

@register_tool(tags=["cloud", "google", "sheets", "create", "data"])
def create_data_spreadsheet(agent: GoogleCloudAgent, title: str, 
                           data: List[List[Any]], headers: List[str] = None) -> Dict[str, Any]:
    """Create spreadsheet with data."""
    # Create spreadsheet
    create_result = agent.execute('create_spreadsheet', title=title)
    
    if create_result['status'] != 'success':
        return create_result
    
    spreadsheet_id = create_result['spreadsheet_id']
    
    # Prepare data with headers
    sheet_data = []
    if headers:
        sheet_data.append(headers)
    sheet_data.extend(data)
    
    # Write data
    write_result = agent.execute('write_range',
                                spreadsheet_id=spreadsheet_id,
                                range_name='Sheet1!A1',
                                values=sheet_data)
    
    return {
        'status': 'success',
        'spreadsheet_id': spreadsheet_id,
        'title': title,
        'rows_written': len(sheet_data),
        'web_view_link': create_result.get('web_view_link')
    }

# High-level wrapper functions for quick cloud operations
@register_tool(tags=["cloud", "google", "drive", "upload", "quick"])
def upload_file_to_drive(credentials_file: str, file_path: str, 
                        parent_folder_id: str = None, file_name: str = None) -> Dict[str, Any]:
    """Quick file upload to Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('upload_file',
                           file_path=file_path,
                           parent_folder_id=parent_folder_id,
                           file_name=file_name)
    except Exception as e:
        return {'status': 'error', 'message': f'Upload failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "drive", "download", "quick"])
def download_file_from_drive(credentials_file: str, file_id: str, 
                            output_path: str = None) -> Dict[str, Any]:
    """Quick file download from Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('download_file',
                           file_id=file_id,
                           output_path=output_path)
    except Exception as e:
        return {'status': 'error', 'message': f'Download failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "docs", "create", "quick"])
def create_google_document(credentials_file: str, title: str, 
                          content: str = None) -> Dict[str, Any]:
    """Quick Google Docs document creation."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('create_document',
                           title=title,
                           content=content)
    except Exception as e:
        return {'status': 'error', 'message': f'Document creation failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "sheets", "create", "quick"])
def create_google_spreadsheet(credentials_file: str, title: str, 
                             sheet_names: List[str] = None) -> Dict[str, Any]:
    """Quick Google Sheets spreadsheet creation."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('create_spreadsheet',
                           title=title,
                           sheet_names=sheet_names)
    except Exception as e:
        return {'status': 'error', 'message': f'Spreadsheet creation failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "drive", "search", "files"])
def search_google_drive(credentials_file: str, query: str, 
                       file_type: str = None) -> Dict[str, Any]:
    """Search files in Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('search_files',
                           query=query,
                           file_type=file_type)
    except Exception as e:
        return {'status': 'error', 'message': f'Search failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "drive", "list", "files"])
def list_google_drive_files(credentials_file: str, folder_id: str = None, 
                           page_size: int = 100) -> Dict[str, Any]:
    """List files in Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('list_files',
                           folder_id=folder_id,
                           page_size=page_size)
    except Exception as e:
        return {'status': 'error', 'message': f'List files failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "drive", "folder", "create"])
def create_google_drive_folder(credentials_file: str, folder_name: str, 
                              parent_folder_id: str = None) -> Dict[str, Any]:
    """Create folder in Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('create_folder',
                           folder_name=folder_name,
                           parent_folder_id=parent_folder_id)
    except Exception as e:
        return {'status': 'error', 'message': f'Folder creation failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "drive", "share", "permissions"])
def share_google_drive_file(credentials_file: str, file_id: str, 
                           email: str = None, role: str = 'reader') -> Dict[str, Any]:
    """Share file in Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('share_file',
                           file_id=file_id,
                           email=email,
                           role=role)
    except Exception as e:
        return {'status': 'error', 'message': f'Share failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "docs", "edit", "text"])
def edit_google_document(credentials_file: str, document_id: str, 
                        old_text: str, new_text: str) -> Dict[str, Any]:
    """Edit text in Google Docs document."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('replace_text',
                           document_id=document_id,
                           old_text=old_text,
                           new_text=new_text)
    except Exception as e:
        return {'status': 'error', 'message': f'Document edit failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "docs", "read", "content"])
def read_google_document(credentials_file: str, document_id: str) -> Dict[str, Any]:
    """Read content from Google Docs document."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('get_document',
                           document_id=document_id,
                           include_content=True)
    except Exception as e:
        return {'status': 'error', 'message': f'Document read failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "sheets", "read", "data"])
def read_google_spreadsheet(credentials_file: str, spreadsheet_id: str, 
                           range_name: str = None) -> Dict[str, Any]:
    """Read data from Google Sheets spreadsheet."""
    try:
        agent = quick_setup(credentials_file)
        if range_name:
            return agent.execute('read_range',
                               spreadsheet_id=spreadsheet_id,
                               range_name=range_name)
        else:
            return agent.execute('get_spreadsheet',
                               spreadsheet_id=spreadsheet_id,
                               include_data=True)
    except Exception as e:
        return {'status': 'error', 'message': f'Spreadsheet read failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "sheets", "write", "data"])
def write_google_spreadsheet(credentials_file: str, spreadsheet_id: str, 
                            range_name: str, values: List[List[Any]]) -> Dict[str, Any]:
    """Write data to Google Sheets spreadsheet."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('write_range',
                           spreadsheet_id=spreadsheet_id,
                           range_name=range_name,
                           values=values)
    except Exception as e:
        return {'status': 'error', 'message': f'Spreadsheet write failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "backup", "download"])
def backup_google_drive(credentials_file: str, backup_folder: str, 
                       file_types: List[str] = None) -> Dict[str, Any]:
    """Backup Google Drive files to local folder."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('backup_drive',
                           backup_folder=backup_folder,
                           file_types=file_types)
    except Exception as e:
        return {'status': 'error', 'message': f'Backup failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "sync", "folder"])
def sync_with_google_drive(credentials_file: str, local_folder: str, 
                          drive_folder_id: str = None) -> Dict[str, Any]:
    """Sync local folder with Google Drive."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('sync_folder',
                           local_folder=local_folder,
                           drive_folder_id=drive_folder_id)
    except Exception as e:
        return {'status': 'error', 'message': f'Sync failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "storage", "usage", "stats"])
def get_google_drive_storage(credentials_file: str) -> Dict[str, Any]:
    """Get Google Drive storage usage information."""
    try:
        agent = quick_setup(credentials_file)
        return agent.execute('get_storage_usage')
    except Exception as e:
        return {'status': 'error', 'message': f'Storage check failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "test", "connection"])
def test_google_cloud_connection(credentials_file: str) -> Dict[str, Any]:
    """Test Google Cloud services connection."""
    try:
        agent = quick_setup(credentials_file)
        # Test connection by getting storage usage
        return agent.execute('get_storage_usage')
    except Exception as e:
        return {'status': 'error', 'message': f'Connection test failed: {str(e)}'}

@register_tool(tags=["cloud", "google", "agent", "create"])
def create_google_cloud_agent() -> GoogleCloudAgent:
    """Create a new Google Cloud services agent."""
    return GoogleCloudAgent()