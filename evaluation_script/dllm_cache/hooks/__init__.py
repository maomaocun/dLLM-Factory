from .cache_hook_LLaDA import register_cache_LLaDA, logout_cache_LLaDA
from .cache_hook_Dream import register_cache_Dream, logout_cache_Dream
from .cache_hook_LLaDA_V import register_cache_LLaDA_V
from .cache_hook_MMaDA import register_cache_MMaDA,logout_cache_MMaDA
__all__ = [
    "register_cache_LLaDA",
    "logout_cache_LLaDA",
    "register_cache_Dream",
    "logout_cache_Dream",
    "register_cache_LLaDA_V",
    "logout_cache_LLaDA_V",
    "register_cache_MMaDA",
    "logout_cache_MMaDA"
]
