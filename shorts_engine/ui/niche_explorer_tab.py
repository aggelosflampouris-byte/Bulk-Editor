import streamlit as st
import logging
from datetime import datetime as _dt, timezone as _tz
from shorts_engine.config import Settings
logger = logging.getLogger(__name__)

def render_niche_explorer_tab(settings: Settings) -> None:
