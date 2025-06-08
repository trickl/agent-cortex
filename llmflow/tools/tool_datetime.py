"""
LLMFlow - A powerful framework for building AI agents based on GAME methodology
(Goals, Actions, Memory, Environment).

DateTime Tool - Handles date and time operations including parsing, formatting, timezone conversions, and schedule management.
"""

import datetime
import json
import logging
from typing import Dict, Any
from llmflow.tools.tool_decorator import register_tool

# Setup a logger for this tool
tool_logger = logging.getLogger(__name__)
# Basic configuration if not already configured by the main application
if not tool_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    tool_logger.addHandler(handler)
    tool_logger.setLevel(logging.DEBUG)

@register_tool(tags=["system", "time", "date"])
def get_current_datetime() -> Dict[str, str]:
    """
    Retrieves the current LOCAL date and time from the system.

    Returns:
        A dictionary containing the current local date and time in various formats,
        including a human-readable string, ISO 8601 format with offset, and timezone information.
        Example (will vary based on system's local time):
        {
            "readable_datetime": "Friday, May 31, 2024, 07:00:00 PM CEST",
            "iso_datetime": "2024-05-31T19:00:00.000000+02:00",
            "date": "2024-05-31",
            "time": "19:00:00.000000",
            "timezone_name": "CEST", // System's local timezone name
            "timezone_offset": "+02:00", // Offset from UTC
            "weekday": "Friday",
            "note": "Provides current date and time based on the system's local settings."
        }
    """
    tool_logger.debug("get_current_datetime tool called (fetching local system time).")
    try:
        # Get current local time from the system
        now_local = datetime.datetime.now().astimezone() # astimezone() ensures tzinfo is populated
        
        tool_logger.debug(f"Raw datetime.datetime.now().astimezone() object: {now_local}")
        tool_logger.debug(f"Raw now_local.year: {now_local.year}, month: {now_local.month}, day: {now_local.day}")
        tool_logger.debug(f"Raw now_local.hour: {now_local.hour}, minute: {now_local.minute}, second: {now_local.second}")
        tool_logger.debug(f"Raw now_local.tzinfo: {now_local.tzinfo}")
        tool_logger.debug(f"Raw now_local.tzname(): {now_local.tzname()}")
        tool_logger.debug(f"Raw now_local.utcoffset(): {now_local.utcoffset()}")

        # Format into human-readable string, %Z should give local timezone name
        readable_dt = now_local.strftime("%A, %B %d, %Y, %I:%M:%S %p %Z")
        tool_logger.debug(f"Formatted readable_dt: {readable_dt}")
        
        iso_dt = now_local.isoformat()
        tool_logger.debug(f"Formatted iso_dt: {iso_dt}")

        timezone_name = now_local.tzname() or "Unknown"
        
        # Format UTC offset
        offset_seconds = now_local.utcoffset().total_seconds() if now_local.utcoffset() else 0
        offset_hours = int(offset_seconds // 3600)
        offset_minutes = int((offset_seconds % 3600) // 60)
        timezone_offset_str = f"{offset_hours:+03d}:{offset_minutes:02d}"

        result = {
            "readable_datetime": readable_dt,
            "iso_datetime": iso_dt,
            "date": now_local.strftime("%Y-%m-%d"),
            "time": now_local.strftime("%H:%M:%S.%f"),
            "timezone_name": timezone_name,
            "timezone_offset_from_utc": timezone_offset_str,
            "weekday": now_local.strftime("%A"),
            "note": "Provides current date and time based on the system's local settings."
        }
        tool_logger.debug(f"Final result dictionary: {json.dumps(result)}")
        return result
    except Exception as e:
        tool_logger.error(f"Error in get_current_datetime (local): {str(e)}", exc_info=True)
        return {
            "error": "Failed to retrieve current local date and time.",
            "details": str(e)
        }

if __name__ == '__main__':
    # Ensure logger for __main__ is also verbose for testing this file directly
    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger("__main__")
    main_logger.info("Testing get_current_datetime (local time) directly...")
    
    current_time_info = get_current_datetime()
    print("\nCurrent Local Date and Time Information (from tool call):")
    print(json.dumps(current_time_info, indent=2))

    # Example specific formats:
    # print(f"Just date: {current_time_info.get('date')}")
    # print(f"Just time: {current_time_info.get('time')}")
    # print(f"ISO Standard: {current_time_info.get('iso_datetime')}") 