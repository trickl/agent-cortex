"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

Calendar Tool - Provides calendar management capabilities including event creation, scheduling, reminders, and calendar synchronization across different platforms.
"""

import os
import json
import time
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict
import uuid
import re
from enum import Enum
import calendar as cal
import pytz
from urllib.parse import urlencode
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Optional imports with fallbacks
try:
    from icalendar import Calendar, Event, vCalAddress, vText, vDDDTypes
    from icalendar.prop import vRecur
    ICALENDAR_AVAILABLE = True
except ImportError:
    ICALENDAR_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False

try:
    import exchangelib
    from exchangelib import Credentials as ExchangeCredentials, Account, CalendarItem
    EXCHANGE_AVAILABLE = True
except ImportError:
    EXCHANGE_AVAILABLE = False

try:
    import caldav
    CALDAV_AVAILABLE = True
except ImportError:
    CALDAV_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class EventStatus(Enum):
    
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"

class RecurrenceFrequency(Enum):
    """Recurrence frequency enumeration."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

class ReminderType(Enum):
    """Reminder type enumeration."""
    POPUP = "popup"
    EMAIL = "email"
    SMS = "sms"

@dataclass
class CalendarConfig:
    """Calendar configuration settings."""
    provider: str = os.getenv('CALENDAR_PROVIDER', 'local')  # google, outlook, caldav, ical, local
    credentials: Dict[str, Any] = None
    timezone: str = os.getenv('CALENDAR_TIMEZONE', 'UTC')
    calendar_id: Optional[str] = os.getenv('CALENDAR_ID', None)
    sync_enabled: bool = os.getenv('CALENDAR_SYNC_ENABLED', 'true').lower() == 'true'
    cache_ttl: int = int(os.getenv('CALENDAR_CACHE_TTL', '300'))  # 5 minutes
    retry_attempts: int = int(os.getenv('CALENDAR_RETRY_ATTEMPTS', '3'))

    def __post_init__(self):
        if self.credentials is None:
            self.credentials = {
                'token_file': os.getenv('CALENDAR_TOKEN_FILE', 'token.json'),
                'credentials_file': os.getenv('CALENDAR_CREDENTIALS_FILE', 'credentials.json'),
                # Exchange/Outlook credentials
                'exchange_email': os.getenv('EXCHANGE_EMAIL', ''),
                'exchange_password': os.getenv('EXCHANGE_PASSWORD', ''),
                'exchange_server': os.getenv('EXCHANGE_SERVER', ''),
                # CalDAV credentials
                'caldav_url': os.getenv('CALDAV_URL', ''),
                'caldav_username': os.getenv('CALDAV_USERNAME', ''),
                'caldav_password': os.getenv('CALDAV_PASSWORD', ''),
                # Local storage
                'storage_dir': os.getenv('CALENDAR_STORAGE_DIR', './calendar_data')
            }

@dataclass
class Reminder:
    """Event reminder configuration."""
    type: ReminderType
    minutes_before: int
    message: Optional[str] = None
    enabled: bool = True

@dataclass
class Recurrence:
    """Event recurrence configuration."""
    frequency: RecurrenceFrequency
    interval: int = 1
    count: Optional[int] = None
    until: Optional[datetime] = None
    by_day: List[str] = None  # ['MO', 'TU', 'WE', 'TH', 'FR', 'SA', 'SU']
    by_month_day: List[int] = None
    by_month: List[int] = None

@dataclass
class Attendee:
    """Event attendee information."""
    email: str
    name: Optional[str] = None
    status: str = 'needsAction'  # needsAction, declined, tentative, accepted
    organizer: bool = False
    optional: bool = False
    comment: Optional[str] = None

@dataclass
class CalendarEvent:
    """Universal calendar event format."""
    id: str
    calendar_id: str
    title: str
    description: Optional[str]
    start_time: datetime
    end_time: datetime
    timezone: str
    location: Optional[str]
    attendees: List[Attendee]
    reminders: List[Reminder]
    recurrence: Optional[Recurrence]
    status: EventStatus
    visibility: str  # default, public, private, confidential
    created_at: datetime
    updated_at: datetime
    organizer: Optional[Attendee]
    metadata: Dict[str, Any]
    all_day: bool = False
    recurring_event_id: Optional[str] = None

@dataclass
class CalendarInfo:
    """Calendar information."""
    id: str
    name: str
    description: Optional[str]
    timezone: str
    color: Optional[str]
    owner: str
    access_role: str  # owner, reader, writer, freeBusyReader
    selected: bool
    primary: bool
    metadata: Dict[str, Any]

class GoogleCalendarIntegration:
    """
    Google Calendar integration using Google Calendar API.
    """
    
    def __init__(self, config: CalendarConfig):
        self.config = config
        self.service = None
        self.credentials = None
        
        if GOOGLE_CALENDAR_AVAILABLE:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar API."""
        try:
            creds = None
            token_file = self.config.credentials.get('token_file', 'token.json')
            credentials_file = self.config.credentials.get('credentials_file', 'credentials.json')
            scopes = ['https://www.googleapis.com/auth/calendar']
            
            # Load existing token
            if os.path.exists(token_file):
                creds = Credentials.from_authorized_user_file(token_file, scopes)
            
            # If there are no valid credentials, get them
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if os.path.exists(credentials_file):
                        flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
                        creds = flow.run_local_server(port=0)
                
                # Save credentials for next run
                with open(token_file, 'w') as token:
                    token.write(creds.to_json())
            
            self.credentials = creds
            self.service = build('calendar', 'v3', credentials=creds)
            
        except Exception as e:
            logging.error(f"Google Calendar authentication failed: {e}")
    
    def get_calendars(self) -> Dict[str, Any]:
        """Get list of calendars."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Service not authenticated'}
            
            calendars_result = self.service.calendarList().list().execute()
            calendars = calendars_result.get('items', [])
            
            calendar_list = []
            for calendar in calendars:
                calendar_list.append(CalendarInfo(
                    id=calendar['id'],
                    name=calendar['summary'],
                    description=calendar.get('description', ''),
                    timezone=calendar.get('timeZone', 'UTC'),
                    color=calendar.get('backgroundColor', ''),
                    owner=calendar.get('primary', False),
                    access_role=calendar.get('accessRole', 'reader'),
                    selected=calendar.get('selected', False),
                    primary=calendar.get('primary', False),
                    metadata=calendar
                ))
            
            return {
                'status': 'success',
                'calendars': [asdict(c) for c in calendar_list],
                'total_calendars': len(calendar_list)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get calendars: {str(e)}'}
    
    def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create new event."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Service not authenticated'}
            
            # Convert to Google Calendar format
            google_event = self._convert_to_google_event(event)
            
            created_event = self.service.events().insert(
                calendarId=event.calendar_id,
                body=google_event
            ).execute()
            
            return {
                'status': 'success',
                'event_id': created_event['id'],
                'event_link': created_event.get('htmlLink', ''),
                'message': f'Event "{event.title}" created successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create event: {str(e)}'}
    
    def update_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Update existing event."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Service not authenticated'}
            
            google_event = self._convert_to_google_event(event)
            
            updated_event = self.service.events().update(
                calendarId=event.calendar_id,
                eventId=event.id,
                body=google_event
            ).execute()
            
            return {
                'status': 'success',
                'event_id': updated_event['id'],
                'message': f'Event "{event.title}" updated successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to update event: {str(e)}'}
    
    def delete_event(self, calendar_id: str, event_id: str) -> Dict[str, Any]:
        """Delete event."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Service not authenticated'}
            
            self.service.events().delete(
                calendarId=calendar_id,
                eventId=event_id
            ).execute()
            
            return {
                'status': 'success',
                'event_id': event_id,
                'message': 'Event deleted successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete event: {str(e)}'}
    
    def get_events(self, calendar_id: str, start_time: datetime = None,
                   end_time: datetime = None, max_results: int = 100) -> Dict[str, Any]:
        """Get events from calendar."""
        try:
            if not self.service:
                return {'status': 'error', 'message': 'Service not authenticated'}
            
            # Set default time range if not provided
            if not start_time:
                start_time = datetime.now(timezone.utc)
            if not end_time:
                end_time = start_time + timedelta(days=30)
            
            # Convert to RFC3339 format
            time_min = start_time.isoformat()
            time_max = end_time.isoformat()
            
            events_result = self.service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            
            event_list = []
            for event in events:
                parsed_event = self._convert_from_google_event(event, calendar_id)
                event_list.append(asdict(parsed_event))
            
            return {
                'status': 'success',
                'events': event_list,
                'total_events': len(event_list),
                'next_page_token': events_result.get('nextPageToken')
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get events: {str(e)}'}
    
    def _convert_to_google_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Convert CalendarEvent to Google Calendar format."""
        google_event = {
            'summary': event.title,
            'description': event.description or '',
            'location': event.location or '',
            'status': event.status.value,
            'visibility': event.visibility
        }
        
        # Set time
        if event.all_day:
            google_event['start'] = {'date': event.start_time.strftime('%Y-%m-%d')}
            google_event['end'] = {'date': event.end_time.strftime('%Y-%m-%d')}
        else:
            google_event['start'] = {
                'dateTime': event.start_time.isoformat(),
                'timeZone': event.timezone
            }
            google_event['end'] = {
                'dateTime': event.end_time.isoformat(),
                'timeZone': event.timezone
            }
        
        # Add attendees
        if event.attendees:
            google_event['attendees'] = []
            for attendee in event.attendees:
                google_attendee = {
                    'email': attendee.email,
                    'responseStatus': attendee.status,
                    'organizer': attendee.organizer,
                    'optional': attendee.optional
                }
                if attendee.name:
                    google_attendee['displayName'] = attendee.name
                if attendee.comment:
                    google_attendee['comment'] = attendee.comment
                
                google_event['attendees'].append(google_attendee)
        
        # Add reminders
        if event.reminders:
            google_event['reminders'] = {
                'useDefault': False,
                'overrides': []
            }
            for reminder in event.reminders:
                if reminder.enabled:
                    google_event['reminders']['overrides'].append({
                        'method': reminder.type.value,
                        'minutes': reminder.minutes_before
                    })
        
        # Add recurrence
        if event.recurrence:
            google_event['recurrence'] = [self._build_rrule(event.recurrence)]
        
        return google_event
    
    def _convert_from_google_event(self, google_event: Dict[str, Any], calendar_id: str) -> CalendarEvent:
        """Convert Google Calendar event to CalendarEvent."""
        # Parse start and end times
        start = google_event.get('start', {})
        end = google_event.get('end', {})
        
        if 'date' in start:
            # All-day event
            start_time = datetime.strptime(start['date'], '%Y-%m-%d')
            end_time = datetime.strptime(end['date'], '%Y-%m-%d')
            all_day = True
            event_timezone = 'UTC'
        else:
            # Timed event
            start_time = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end['dateTime'].replace('Z', '+00:00'))
            all_day = False
            event_timezone = start.get('timeZone', 'UTC')
        
        # Parse attendees
        attendees = []
        for attendee_data in google_event.get('attendees', []):
            attendees.append(Attendee(
                email=attendee_data['email'],
                name=attendee_data.get('displayName'),
                status=attendee_data.get('responseStatus', 'needsAction'),
                organizer=attendee_data.get('organizer', False),
                optional=attendee_data.get('optional', False),
                comment=attendee_data.get('comment')
            ))
        
        # Parse reminders
        reminders = []
        reminder_data = google_event.get('reminders', {})
        if not reminder_data.get('useDefault', True):
            for override in reminder_data.get('overrides', []):
                reminders.append(Reminder(
                    type=ReminderType(override['method']),
                    minutes_before=override['minutes'],
                    enabled=True
                ))
        
        # Parse organizer
        organizer = None
        organizer_data = google_event.get('organizer', {})
        if organizer_data:
            organizer = Attendee(
                email=organizer_data['email'],
                name=organizer_data.get('displayName'),
                organizer=True
            )
        
        return CalendarEvent(
            id=google_event['id'],
            calendar_id=calendar_id,
            title=google_event.get('summary', 'No Title'),
            description=google_event.get('description'),
            start_time=start_time,
            end_time=end_time,
            timezone=event_timezone,
            location=google_event.get('location'),
            attendees=attendees,
            reminders=reminders,
            recurrence=self._parse_recurrence(google_event.get('recurrence', [])),
            status=EventStatus(google_event.get('status', 'confirmed')),
            visibility=google_event.get('visibility', 'default'),
            created_at=datetime.fromisoformat(google_event['created'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(google_event['updated'].replace('Z', '+00:00')),
            organizer=organizer,
            metadata=google_event,
            all_day=all_day,
            recurring_event_id=google_event.get('recurringEventId')
        )
    
    def _build_rrule(self, recurrence: Recurrence) -> str:
        """Build RRULE string from recurrence."""
        rule_parts = [f'FREQ={recurrence.frequency.value.upper()}']
        
        if recurrence.interval > 1:
            rule_parts.append(f'INTERVAL={recurrence.interval}')
        
        if recurrence.count:
            rule_parts.append(f'COUNT={recurrence.count}')
        elif recurrence.until:
            rule_parts.append(f'UNTIL={recurrence.until.strftime("%Y%m%dT%H%M%SZ")}')
        
        if recurrence.by_day:
            rule_parts.append(f'BYDAY={",".join(recurrence.by_day)}')
        
        if recurrence.by_month_day:
            rule_parts.append(f'BYMONTHDAY={",".join(map(str, recurrence.by_month_day))}')
        
        if recurrence.by_month:
            rule_parts.append(f'BYMONTH={",".join(map(str, recurrence.by_month))}')
        
        return f'RRULE:{";".join(rule_parts)}'
    
    def _parse_recurrence(self, recurrence_rules: List[str]) -> Optional[Recurrence]:
        """Parse recurrence rules."""
        if not recurrence_rules:
            return None
        
        # Simple RRULE parsing (can be extended)
        rule = recurrence_rules[0]
        if rule.startswith('RRULE:'):
            rule = rule[6:]  # Remove 'RRULE:' prefix
        
        parts = rule.split(';')
        freq = None
        interval = 1
        count = None
        until = None
        by_day = None
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                if key == 'FREQ':
                    freq = RecurrenceFrequency(value.lower())
                elif key == 'INTERVAL':
                    interval = int(value)
                elif key == 'COUNT':
                    count = int(value)
                elif key == 'UNTIL':
                    until = datetime.strptime(value, '%Y%m%dT%H%M%SZ')
                elif key == 'BYDAY':
                    by_day = value.split(',')
        
        if freq:
            return Recurrence(
                frequency=freq,
                interval=interval,
                count=count,
                until=until,
                by_day=by_day
            )
        
        return None

class LocalCalendarStorage:
    """
    Local calendar storage using JSON files.
    """
    
    def __init__(self, config: CalendarConfig):
        self.config = config
        self.storage_dir = config.credentials.get('storage_dir', './calendar_data')
        self.calendars_file = os.path.join(self.storage_dir, 'calendars.json')
        self.events_file = os.path.join(self.storage_dir, 'events.json')
        
        # Create storage directory
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage files."""
        if not os.path.exists(self.calendars_file):
            with open(self.calendars_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.events_file):
            with open(self.events_file, 'w') as f:
                json.dump({}, f)
    
    def _load_calendars(self) -> Dict[str, Any]:
        """Load calendars from storage."""
        try:
            with open(self.calendars_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_calendars(self, calendars: Dict[str, Any]):
        """Save calendars to storage."""
        with open(self.calendars_file, 'w') as f:
            json.dump(calendars, f, indent=2, default=str)
    
    def _load_events(self) -> Dict[str, Any]:
        """Load events from storage."""
        try:
            with open(self.events_file, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_events(self, events: Dict[str, Any]):
        """Save events to storage."""
        with open(self.events_file, 'w') as f:
            json.dump(events, f, indent=2, default=str)
    
    def get_calendars(self) -> Dict[str, Any]:
        """Get list of calendars."""
        try:
            calendars_data = self._load_calendars()
            
            calendar_list = []
            for calendar_id, calendar_data in calendars_data.items():
                calendar_list.append(CalendarInfo(
                    id=calendar_id,
                    name=calendar_data['name'],
                    description=calendar_data.get('description', ''),
                    timezone=calendar_data.get('timezone', 'UTC'),
                    color=calendar_data.get('color', '#3F51B5'),
                    owner=calendar_data.get('owner', 'local'),
                    access_role='owner',
                    selected=calendar_data.get('selected', True),
                    primary=calendar_data.get('primary', False),
                    metadata=calendar_data
                ))
            
            return {
                'status': 'success',
                'calendars': [asdict(c) for c in calendar_list],
                'total_calendars': len(calendar_list)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get calendars: {str(e)}'}
    
    def create_calendar(self, name: str, description: str = '', timezone: str = 'UTC',
                       color: str = '#3F51B5') -> Dict[str, Any]:
        """Create new calendar."""
        try:
            calendars = self._load_calendars()
            
            calendar_id = str(uuid.uuid4())
            calendars[calendar_id] = {
                'name': name,
                'description': description,
                'timezone': timezone,
                'color': color,
                'owner': 'local',
                'selected': True,
                'primary': len(calendars) == 0,
                'created_at': datetime.now().isoformat()
            }
            
            self._save_calendars(calendars)
            
            return {
                'status': 'success',
                'calendar_id': calendar_id,
                'message': f'Calendar "{name}" created successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create calendar: {str(e)}'}
    
    def create_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create new event."""
        try:
            events = self._load_events()
            
            if not event.id:
                event.id = str(uuid.uuid4())
            
            # Convert datetime objects to ISO format for JSON storage
            event_data = asdict(event)
            event_data['start_time'] = event.start_time.isoformat()
            event_data['end_time'] = event.end_time.isoformat()
            event_data['created_at'] = event.created_at.isoformat()
            event_data['updated_at'] = event.updated_at.isoformat()
            
            events[event.id] = event_data
            self._save_events(events)
            
            return {
                'status': 'success',
                'event_id': event.id,
                'message': f'Event "{event.title}" created successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create event: {str(e)}'}
    
    def update_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Update existing event."""
        try:
            events = self._load_events()
            
            if event.id not in events:
                return {'status': 'error', 'message': f'Event {event.id} not found'}
            
            # Update timestamp
            event.updated_at = datetime.now()
            
            # Convert datetime objects to ISO format
            event_data = asdict(event)
            event_data['start_time'] = event.start_time.isoformat()
            event_data['end_time'] = event.end_time.isoformat()
            event_data['created_at'] = event.created_at.isoformat()
            event_data['updated_at'] = event.updated_at.isoformat()
            
            events[event.id] = event_data
            self._save_events(events)
            
            return {
                'status': 'success',
                'event_id': event.id,
                'message': f'Event "{event.title}" updated successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to update event: {str(e)}'}
    
    def delete_event(self, calendar_id: str, event_id: str) -> Dict[str, Any]:
        """Delete event."""
        try:
            events = self._load_events()
            
            if event_id not in events:
                return {'status': 'error', 'message': f'Event {event_id} not found'}
            
            del events[event_id]
            self._save_events(events)
            
            return {
                'status': 'success',
                'event_id': event_id,
                'message': 'Event deleted successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete event: {str(e)}'}
    
    def get_events(self, calendar_id: str, start_time: datetime = None,
                   end_time: datetime = None, max_results: int = 100) -> Dict[str, Any]:
        """Get events from calendar."""
        try:
            events_data = self._load_events()
            
            # Filter events by calendar and time range
            filtered_events = []
            for event_id, event_data in events_data.items():
                if event_data['calendar_id'] != calendar_id:
                    continue
                
                event_start = datetime.fromisoformat(event_data['start_time'])
                event_end = datetime.fromisoformat(event_data['end_time'])
                
                # Check time range
                if start_time and event_end < start_time:
                    continue
                if end_time and event_start > end_time:
                    continue
                
                # Convert back to CalendarEvent format
                event_data['start_time'] = event_start
                event_data['end_time'] = event_end
                event_data['created_at'] = datetime.fromisoformat(event_data['created_at'])
                event_data['updated_at'] = datetime.fromisoformat(event_data['updated_at'])
                
                filtered_events.append(event_data)
                
                if len(filtered_events) >= max_results:
                    break
            
            # Sort by start time
            filtered_events.sort(key=lambda x: x['start_time'])
            
            return {
                'status': 'success',
                'events': filtered_events,
                'total_events': len(filtered_events)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get events: {str(e)}'}

class CalendarManagementTool:
    """
    Comprehensive calendar management tool supporting multiple providers.
    """
    
    def __init__(self):
        self.providers = {}
        self.configs = {}
        self.default_timezone = 'UTC'
        self.operation_history = []
        
        # Statistics
        self.stats = {
            'events_created': 0,
            'events_updated': 0,
            'events_deleted': 0,
            'calendars_managed': 0,
            'last_activity': None
        }
        
        # Event cache
        self.event_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def add_provider(self, provider_name: str, config: CalendarConfig) -> Dict[str, Any]:
        """Add calendar provider."""
        try:
            self.configs[provider_name] = config
            
            if config.provider == 'google':
                if not GOOGLE_CALENDAR_AVAILABLE:
                    return {'status': 'error', 'message': 'Google Calendar API not available'}
                self.providers[provider_name] = GoogleCalendarIntegration(config)
            elif config.provider == 'local':
                self.providers[provider_name] = LocalCalendarStorage(config)
            else:
                return {'status': 'error', 'message': f'Unsupported provider: {config.provider}'}
            
            return {
                'status': 'success',
                'provider': config.provider,
                'provider_name': provider_name,
                'message': f'Provider {provider_name} ({config.provider}) added successfully'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to add provider: {str(e)}'}
    
    def create_event(self, provider_name: str, calendar_id: str, title: str,
                    start_time: datetime, end_time: datetime, **kwargs) -> Dict[str, Any]:
        """Create calendar event."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            # Create CalendarEvent object
            event = CalendarEvent(
                id=kwargs.get('event_id', str(uuid.uuid4())),
                calendar_id=calendar_id,
                title=title,
                description=kwargs.get('description'),
                start_time=start_time,
                end_time=end_time,
                timezone=kwargs.get('timezone', self.default_timezone),
                location=kwargs.get('location'),
                attendees=[],
                reminders=[],
                recurrence=kwargs.get('recurrence'),
                status=EventStatus(kwargs.get('status', 'confirmed')),
                visibility=kwargs.get('visibility', 'default'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                organizer=kwargs.get('organizer'),
                metadata=kwargs.get('metadata', {}),
                all_day=kwargs.get('all_day', False)
            )
            
            # Add attendees if provided
            if 'attendees' in kwargs:
                for attendee_data in kwargs['attendees']:
                    attendee = Attendee(**attendee_data)
                    event.attendees.append(attendee)
            
            # Add reminders if provided
            if 'reminders' in kwargs:
                for reminder_data in kwargs['reminders']:
                    if isinstance(reminder_data, dict):
                        reminder = Reminder(
                            type=ReminderType(reminder_data['type']),
                            minutes_before=reminder_data['minutes_before'],
                            message=reminder_data.get('message'),
                            enabled=reminder_data.get('enabled', True)
                        )
                    else:
                        reminder = reminder_data
                    event.reminders.append(reminder)
            
            # Create event using provider
            provider = self.providers[provider_name]
            result = provider.create_event(event)
            
            if result['status'] == 'success':
                self.stats['events_created'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
                
                # Record operation
                self.operation_history.append({
                    'type': 'create_event',
                    'provider': provider_name,
                    'calendar_id': calendar_id,
                    'event_title': title,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                })
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create event: {str(e)}'}
    
    def update_event(self, provider_name: str, event_id: str, **updates) -> Dict[str, Any]:
        """Update existing event."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            provider = self.providers[provider_name]
            
            # Get existing event first
            # This is a simplified approach - in practice, you'd need to implement get_event
            # For now, we'll create a new event object with updates
            
            # Create updated event object (simplified)
            event_data = {
                'id': event_id,
                'updated_at': datetime.now(),
                **updates
            }
            
            # Convert to CalendarEvent if needed
            if 'calendar_id' in updates and 'title' in updates:
                event = CalendarEvent(**event_data)
                result = provider.update_event(event)
            else:
                return {'status': 'error', 'message': 'Insufficient event data for update'}
            
            if result['status'] == 'success':
                self.stats['events_updated'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to update event: {str(e)}'}
    
    def delete_event(self, provider_name: str, calendar_id: str, event_id: str) -> Dict[str, Any]:
        """Delete event."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            provider = self.providers[provider_name]
            result = provider.delete_event(calendar_id, event_id)
            
            if result['status'] == 'success':
                self.stats['events_deleted'] += 1
                self.stats['last_activity'] = datetime.now().isoformat()
                
                # Clear from cache
                cache_key = f"{provider_name}:{calendar_id}:{event_id}"
                if cache_key in self.event_cache:
                    del self.event_cache[cache_key]
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to delete event: {str(e)}'}
    
    def get_events(self, provider_name: str, calendar_id: str,
                   start_date: Union[str, datetime] = None,
                   end_date: Union[str, datetime] = None,
                   max_results: int = 100) -> Dict[str, Any]:
        """Get events from calendar."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            # Convert string dates to datetime
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            
            provider = self.providers[provider_name]
            result = provider.get_events(calendar_id, start_date, end_date, max_results)
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get events: {str(e)}'}
    
    def search_events(self, provider_name: str, calendar_id: str, query: str,
                     start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Search events by text query."""
        try:
            # Get all events in range
            events_result = self.get_events(provider_name, calendar_id, start_date, end_date)
            
            if events_result['status'] != 'success':
                return events_result
            
            # Filter events by query
            matching_events = []
            query_lower = query.lower()
            
            for event in events_result['events']:
                if (query_lower in event['title'].lower() or
                    (event['description'] and query_lower in event['description'].lower()) or
                    (event['location'] and query_lower in event['location'].lower())):
                    matching_events.append(event)
            
            return {
                'status': 'success',
                'events': matching_events,
                'total_events': len(matching_events),
                'query': query
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Search failed: {str(e)}'}
    
    def get_calendars(self, provider_name: str) -> Dict[str, Any]:
        """Get calendars from provider."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            provider = self.providers[provider_name]
            return provider.get_calendars()
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to get calendars: {str(e)}'}
    
    def create_calendar(self, provider_name: str, name: str, **kwargs) -> Dict[str, Any]:
        """Create new calendar."""
        try:
            if provider_name not in self.providers:
                return {'status': 'error', 'message': f'Provider {provider_name} not configured'}
            
            provider = self.providers[provider_name]
            
            if hasattr(provider, 'create_calendar'):
                result = provider.create_calendar(name, **kwargs)
                
                if result['status'] == 'success':
                    self.stats['calendars_managed'] += 1
                
                return result
            else:
                return {'status': 'error', 'message': 'Provider does not support calendar creation'}
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create calendar: {str(e)}'}
    
    def find_free_time(self, provider_name: str, calendar_ids: List[str],
                      start_date: datetime, end_date: datetime,
                      duration_minutes: int = 60, working_hours: Tuple[int, int] = (9, 17)) -> Dict[str, Any]:
        """Find free time slots across multiple calendars."""
        try:
            all_events = []
            
            # Get events from all calendars
            for calendar_id in calendar_ids:
                events_result = self.get_events(provider_name, calendar_id, start_date, end_date)
                if events_result['status'] == 'success':
                    all_events.extend(events_result['events'])
            
            # Sort events by start time
            all_events.sort(key=lambda x: x['start_time'])
            
            # Find free slots
            free_slots = []
            current_time = start_date
            
            while current_time < end_date:
                # Check if current time is within working hours
                if working_hours[0] <= current_time.hour < working_hours[1]:
                    slot_end = current_time + timedelta(minutes=duration_minutes)
                    
                    # Check for conflicts
                    conflict = False
                    for event in all_events:
                        event_start = datetime.fromisoformat(event['start_time']) if isinstance(event['start_time'], str) else event['start_time']
                        event_end = datetime.fromisoformat(event['end_time']) if isinstance(event['end_time'], str) else event['end_time']
                        
                        if (current_time < event_end and slot_end > event_start):
                            conflict = True
                            current_time = event_end
                            break
                    
                    if not conflict:
                        free_slots.append({
                            'start_time': current_time.isoformat(),
                            'end_time': slot_end.isoformat(),
                            'duration_minutes': duration_minutes
                        })
                        current_time = slot_end
                    
                else:
                    # Skip to next working hour
                    if current_time.hour < working_hours[0]:
                        current_time = current_time.replace(hour=working_hours[0], minute=0, second=0)
                    else:
                        current_time = (current_time + timedelta(days=1)).replace(hour=working_hours[0], minute=0, second=0)
            
            return {
                'status': 'success',
                'free_slots': free_slots,
                'total_slots': len(free_slots),
                'duration_minutes': duration_minutes,
                'working_hours': working_hours
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to find free time: {str(e)}'}
    
    def create_recurring_event(self, provider_name: str, calendar_id: str, title: str,
                             start_time: datetime, end_time: datetime, frequency: str,
                             **kwargs) -> Dict[str, Any]:
        """Create recurring event."""
        try:
            # Create recurrence object
            recurrence = Recurrence(
                frequency=RecurrenceFrequency(frequency),
                interval=kwargs.get('interval', 1),
                count=kwargs.get('count'),
                until=kwargs.get('until'),
                by_day=kwargs.get('by_day'),
                by_month_day=kwargs.get('by_month_day'),
                by_month=kwargs.get('by_month')
            )
            
            return self.create_event(
                provider_name=provider_name,
                calendar_id=calendar_id,
                title=title,
                start_time=start_time,
                end_time=end_time,
                recurrence=recurrence,
                **kwargs
            )
            
        except Exception as e:
            return {'status': 'error', 'message': f'Failed to create recurring event: {str(e)}'}
    
    def export_calendar(self, provider_name: str, calendar_id: str,
                       format: str = 'ical', file_path: str = None,
                       start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Export calendar to file."""
        try:
            if not ICALENDAR_AVAILABLE and format == 'ical':
                return {'status': 'error', 'message': 'iCalendar library not available'}
            
            # Get events
            events_result = self.get_events(provider_name, calendar_id, start_date, end_date)
            if events_result['status'] != 'success':
                return events_result
            
            # Generate filename if not provided
            if not file_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_path = f'calendar_export_{calendar_id}_{timestamp}.{format}'
            
            if format == 'ical':
                # Create iCalendar
                cal = Calendar()
                cal.add('prodid', '-//Calendar Management Tool//EN')
                cal.add('version', '2.0')
                
                for event_data in events_result['events']:
                    event = Event()
                    event.add('summary', event_data['title'])
                    event.add('dtstart', datetime.fromisoformat(event_data['start_time']) if isinstance(event_data['start_time'], str) else event_data['start_time'])
                    event.add('dtend', datetime.fromisoformat(event_data['end_time']) if isinstance(event_data['end_time'], str) else event_data['end_time'])
                    
                    if event_data['description']:
                        event.add('description', event_data['description'])
                    if event_data['location']:
                        event.add('location', event_data['location'])
                    
                    event.add('uid', event_data['id'])
                    cal.add_component(event)
                
                # Write to file
                with open(file_path, 'wb') as f:
                    f.write(cal.to_ical())
            
            elif format == 'json':
                # Export as JSON
                with open(file_path, 'w') as f:
                    json.dump(events_result['events'], f, indent=2, default=str)
            
            elif format == 'csv' and PANDAS_AVAILABLE:
                # Export as CSV
                df = pd.DataFrame(events_result['events'])
                df.to_csv(file_path, index=False)
            
            else:
                return {'status': 'error', 'message': f'Unsupported export format: {format}'}
            
            return {
                'status': 'success',
                'file_path': file_path,
                'format': format,
                'events_exported': len(events_result['events']),
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Export failed: {str(e)}'}
    
    def import_calendar(self, provider_name: str, calendar_id: str,
                       file_path: str, format: str = 'ical') -> Dict[str, Any]:
        """Import calendar from file."""
        try:
            if not os.path.exists(file_path):
                return {'status': 'error', 'message': f'File not found: {file_path}'}
            
            imported_events = 0
            failed_imports = 0
            
            if format == 'ical':
                if not ICALENDAR_AVAILABLE:
                    return {'status': 'error', 'message': 'iCalendar library not available'}
                
                with open(file_path, 'rb') as f:
                    cal = Calendar.from_ical(f.read())
                
                for component in cal.walk():
                    if component.name == "VEVENT":
                        try:
                            # Extract event data
                            title = str(component.get('summary', 'No Title'))
                            start_time = component.get('dtstart').dt
                            end_time = component.get('dtend').dt
                            description = str(component.get('description', ''))
                            location = str(component.get('location', ''))
                            
                            # Convert date to datetime if needed
                            if isinstance(start_time, datetime.date) and not isinstance(start_time, datetime.datetime):
                                start_time = datetime.combine(start_time, datetime.min.time())
                            if isinstance(end_time, datetime.date) and not isinstance(end_time, datetime.datetime):
                                end_time = datetime.combine(end_time, datetime.min.time())
                            
                            # Create event
                            result = self.create_event(
                                provider_name=provider_name,
                                calendar_id=calendar_id,
                                title=title,
                                start_time=start_time,
                                end_time=end_time,
                                description=description,
                                location=location
                            )
                            
                            if result['status'] == 'success':
                                imported_events += 1
                            else:
                                failed_imports += 1
                                
                        except Exception as e:
                            failed_imports += 1
                            logging.warning(f"Failed to import event: {e}")
            
            elif format == 'json':
                with open(file_path, 'r') as f:
                    events_data = json.load(f)
                
                for event_data in events_data:
                    try:
                        result = self.create_event(
                            provider_name=provider_name,
                            calendar_id=calendar_id,
                            **event_data
                        )
                        
                        if result['status'] == 'success':
                            imported_events += 1
                        else:
                            failed_imports += 1
                            
                    except Exception as e:
                        failed_imports += 1
                        logging.warning(f"Failed to import event: {e}")
            
            else:
                return {'status': 'error', 'message': f'Unsupported import format: {format}'}
            
            return {
                'status': 'success',
                'imported_events': imported_events,
                'failed_imports': failed_imports,
                'total_processed': imported_events + failed_imports,
                'file_path': file_path,
                'format': format
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f'Import failed: {str(e)}'}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            'status': 'success',
            'stats': self.stats,
            'providers': list(self.providers.keys()),
            'operation_history_count': len(self.operation_history),
            'cache_size': len(self.event_cache),
            'available_integrations': {
                'google_calendar': GOOGLE_CALENDAR_AVAILABLE,
                'icalendar': ICALENDAR_AVAILABLE,
                'exchange': EXCHANGE_AVAILABLE,
                'caldav': CALDAV_AVAILABLE
            }
        }

# Agent framework integration
class CalendarAgent:
    """
    Agent wrapper for the calendar management tool.
    """
    
    def __init__(self):
        self.tool = CalendarManagementTool()
        self.capabilities = [
            'add_provider',
            'create_event',
            'update_event',
            'delete_event',
            'get_events',
            'search_events',
            'get_calendars',
            'create_calendar',
            'find_free_time',
            'create_recurring_event',
            'export_calendar',
            'import_calendar',
            'get_stats'
        ]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific calendar operation."""
        try:
            if action == 'add_provider':
                return self.tool.add_provider(**kwargs)
            elif action == 'create_event':
                return self.tool.create_event(**kwargs)
            elif action == 'update_event':
                return self.tool.update_event(**kwargs)
            elif action == 'delete_event':
                return self.tool.delete_event(**kwargs)
            elif action == 'get_events':
                return self.tool.get_events(**kwargs)
            elif action == 'search_events':
                return self.tool.search_events(**kwargs)
            elif action == 'get_calendars':
                return self.tool.get_calendars(**kwargs)
            elif action == 'create_calendar':
                return self.tool.create_calendar(**kwargs)
            elif action == 'find_free_time':
                return self.tool.find_free_time(**kwargs)
            elif action == 'create_recurring_event':
                return self.tool.create_recurring_event(**kwargs)
            elif action == 'export_calendar':
                return self.tool.export_calendar(**kwargs)
            elif action == 'import_calendar':
                return self.tool.import_calendar(**kwargs)
            elif action == 'get_stats':
                return self.tool.get_comprehensive_stats()
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
                
        except Exception as e:
            return {'status': 'error', 'message': f'Error executing {action}: {str(e)}'}
    
    def get_capabilities(self) -> List[str]:
        """Return list of available capabilities."""
        return self.capabilities.copy()

# Quick utility functions
def quick_google_calendar_setup(credentials_file: str, token_file: str = 'token.json') -> CalendarConfig:
    """Quick Google Calendar setup."""
    return CalendarConfig(
        provider='google',
        credentials={
            'credentials_file': credentials_file,
            'token_file': token_file
        },
        timezone='UTC'
    )

def quick_local_calendar_setup(storage_dir: str = './calendar_data') -> CalendarConfig:
    """Quick local calendar setup."""
    return CalendarConfig(
        provider='local',
        credentials={
            'storage_dir': storage_dir
        },
        timezone='UTC'
    )

def create_simple_event(title: str, start_time: datetime, duration_hours: int = 1,
                       description: str = None, location: str = None) -> Dict[str, Any]:
    """Create simple event data structure."""
    end_time = start_time + timedelta(hours=duration_hours)
    
    return {
        'title': title,
        'start_time': start_time,
        'end_time': end_time,
        'description': description,
        'location': location,
        'all_day': False
    }

def create_reminder(minutes_before: int, reminder_type: str = 'popup') -> Reminder:
    """Create reminder object."""
    return Reminder(
        type=ReminderType(reminder_type),
        minutes_before=minutes_before,
        enabled=True
    )

def create_weekly_recurrence(count: int = None, until: datetime = None) -> Recurrence:
    """Create weekly recurrence."""
    return Recurrence(
        frequency=RecurrenceFrequency.WEEKLY,
        interval=1,
        count=count,
        until=until
    )