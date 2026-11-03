"""Safe structured operational events."""

import logging
from datetime import datetime, timezone

from aeris.domain.schemas import StructuredEvent

logger = logging.getLogger("aeris.events")


def emit_event(event: StructuredEvent) -> None:
    logger.info(event.event, extra={"event_data": event.model_dump(mode="json")})


def event(name: str, **fields: object) -> StructuredEvent:
    return StructuredEvent(event=name, timestamp=datetime.now(timezone.utc), **fields)
