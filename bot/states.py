from aiogram.fsm.state import State, StatesGroup


class EditingState(StatesGroup):
    """States for Gemini-powered image / video editing."""

    WAITING_IMAGE_COMMAND = State()
    WAITING_VIDEO_COMMAND = State()


class GenerationState(StatesGroup):
    """States for the video generation process"""

    SET_SYSTEM_IMAGE_PROMPT = State()
    SET_SYSTEM_PLOT_PROMPT = State()
    SET_SYSTEM_VIDEO_GENERATION_PROMPT = State()
    ADD_CHAT_ACCOUNT = State()
    SET_NEGATIVE_PROMPT = State()
    SET_GRADE_PARAMS = State()
    SET_VOICE_ID = State()
    SET_MUSIC_PATH = State()
    SET_UTC_OFFSET = State()

    RAW_PROMPT = State()
    ENHANCE_PROMPT = State()
    CONFIRM_IMAGE = State()
    CONFIRM_VIDEO_PROMPT = State()
    EDIT_SCENE = State()
    EDIT_VOICEOVER = State()
    CONFIRM_VIDEO = State()
    VIDEO_EDIT_PROMPT = State()
    CONFIRM_PUBLISH = State()
    SET_VIDEO_TITLE = State()
    SELECT_PUBLISH_TIME = State()
    SET_PUBLISH_TIME = State()
