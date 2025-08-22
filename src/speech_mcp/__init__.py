import argparse
import sys
import os
import signal
import faulthandler
from .server import mcp
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__)

# Enable faulthandler to help debug segfaults and deadlocks
faulthandler.enable()

# Handle signals to ensure clean shutdown
def signal_handler(sig, frame):
    # Log the signal
    logger.info(f"Received signal {sig}, shutting down...")
    
    # Dump stack traces to help identify where threads might be stuck
    faulthandler.dump_traceback(file=sys.stderr)
    
    # Exit
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Speech MCP: Voice interaction with speech recognition."""
    try:
        # Check if stdin/stdout are available
        if sys.stdin is None or sys.stdout is None or sys.stdin.closed or sys.stdout.closed:
            print("Error: stdin or stdout is closed or not available, cannot run MCP server")
            
            # Create a special file to indicate the error
            try:
                with open(os.path.expanduser("~/.speech-mcp/startup_error.log"), "w") as f:
                    f.write("Error: stdin or stdout is closed or not available, cannot run MCP server")
            except Exception:
                pass
                
            sys.exit(1)
            
        logger.info("Starting Speech MCP server...")
        
        parser = argparse.ArgumentParser(
            description="Voice interaction with speech recognition."
        )
        parser.parse_args()
        
        # SSE sidecar eager startup removed; MCP-UI host manages UI state
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