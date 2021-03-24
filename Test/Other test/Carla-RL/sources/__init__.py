from .agent import AGENT_IMAGE_TYPE
from . import models

import settings
settings.AGENT_IMG_TYPE = getattr(AGENT_IMAGE_TYPE, settings.AGENT_IMG_TYPE)
if models.MODEL_NAME_PREFIX:
    settings.MODEL_NAME = models.MODEL_NAME_PREFIX + ('_' if models.MODEL_NAME else '') + settings.MODEL_NAME
