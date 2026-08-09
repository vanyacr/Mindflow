"""Background keystroke collection service for continuous baseline learning."""
import time
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
log_dir = Path("data/keystroke")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "background_service.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Graceful shutdown flag
SHUTDOWN_EVENT = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_EVENT
    logger.info("Shutdown signal received, stopping background service...")
    SHUTDOWN_EVENT = True

def main():
    """Main background service loop"""
    from keystroke import KeystrokeTracker
    
    logger.info("=" * 60)
    logger.info("KEYSTROKE BACKGROUND SERVICE STARTED")
    logger.info("=" * 60)
    logger.info(f"Baseline file: data/keystroke/baseline_auto.json")
    logger.info(f"Log file: {log_file}")
    logger.info("Service will run continuously, collecting keystroke data...")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize tracker
    tracker = KeystrokeTracker(auto_load_baseline=True, poll_interval_s=0.1)
    
    try:
        # Start listening
        tracker.start_listener()
        logger.info("Keystroke listener started")
        
        # Collection parameters
        window_seconds = 300  # Analyze last 5 minutes of typing activity
        update_interval_seconds = 300  # Run one analysis every 5 minutes
        last_update_time = time.time()
        
        collection_number = 0
        
        while not SHUTDOWN_EVENT:
            try:
                # Wait for update interval
                time.sleep(30)  # Check every 30 seconds
                
                current_time = time.time()
                
                # Every 5 minutes, compute score and potentially update baseline
                if (current_time - last_update_time) >= update_interval_seconds:
                    collection_number += 1
                    logger.info(f"\n--- Collection #{collection_number} ---")
                    logger.info("Computing keystroke stress from last 5 minutes of typing...")
                    
                    # Get stress score
                    result = tracker.keystroke_stress_score(window_seconds=window_seconds)
                    
                    keystroke_score = result.get('keystroke_score', 0)
                    sample_count = result.get('features', {}).get('sample_count', 0)
                    baseline_updated = result.get('baseline_updated')
                    model_used = result.get('model_used', 'unknown')
                    
                    logger.info(f"Keystroke score: {keystroke_score:.4f}")
                    logger.info(f"Samples collected: {sample_count}")
                    logger.info(f"Model used: {model_used}")
                    
                    # Log baseline update if it happened
                    if baseline_updated:
                        logger.info("BASELINE UPDATED - Personal profile refined")
                        logger.info(f"Baseline update count: {result.get('baseline_update_count', 0)}")
                    else:
                        if sample_count < 8:
                            logger.info(f"Insufficient samples ({sample_count} < 8) for baseline update")
                        elif keystroke_score > 0.45:
                            logger.info(f"Typing detected as stressed ({keystroke_score:.2f}) - skipping baseline update")
                        else:
                            logger.info("No baseline update needed this window")
                    
                    # Log timestamp
                    logger.info(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info("-" * 40)
                    
                    last_update_time = current_time
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error during collection: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error(f"Fatal error in background service: {e}", exc_info=True)
        sys.exit(1)
    finally:
        try:
            tracker.stop_listener()
            logger.info("Keystroke listener stopped")
        except:
            pass
        logger.info("=" * 60)
        logger.info("KEYSTROKE BACKGROUND SERVICE STOPPED")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()
