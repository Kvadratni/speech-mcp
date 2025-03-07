import argparse
import sys
import os
import signal
import atexit
import faulthandler
from .server import mcp, cleanup_ui_process
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

# Enable faulthandler to help debug segfaults and deadlocks
faulthandler.enable()

# Ensure UI process is cleaned up on exit
atexit.register(cleanup_ui_process)

# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    # Log the signal
    logger.info(f"Received signal {sig}, shutting down...")
    
    # Dump stack traces to help identify where threads might be stuck
    faulthandler.dump_traceback(file=sys.stderr)
    
    # Clean up
    cleanup_ui_process()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Speech MCP: Voice interaction with speech recognition."""
    logger.info("Starting Speech MCP server...")
    
    try:
        parser = argparse.ArgumentParser(
            description="Voice interaction with speech recognition."
        )
        parser.parse_args()
        
        logger.info("Running MCP server...")
        mcp.run()
    except Exception as e:
        # Log the exception
        logger.exception(f"Error running MCP server: {e}")
        
        # Dump stack traces on unhandled exceptions as well
        faulthandler.dump_traceback(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()