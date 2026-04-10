import logging
import json
import os
from datetime import datetime

# Create logs directory FIRST, before basicConfig tries to open the file
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/rag.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log_pipeline(query, stages, response, status="success", error=None):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "stages_ms": stages,
        "total_ms": sum(stages.values()),
        "response_preview": response[:100] if response else None,
        "status": status,
        "error": error,
    }
    if status == "success":
        logging.info(json.dumps(entry))
    else:
        logging.error(json.dumps(entry))
    return entry