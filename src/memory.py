
import aiosqlite
from logger import logger

DB_PATH = "voice_agent.db"

async def init_memory():
    """Initialize the database and tables for long-term memory."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    preferences TEXT
                )
            """)
            await db.commit()
        logger.info(f"Initialized memory database at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize memory: {e}")
        raise


async def get_user_profile(user_id: str) -> str:
    """Retrieve user profile preferences."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT preferences FROM user_profiles WHERE user_id = ?", (user_id,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else ""
    except Exception as e:
        logger.error(f"Error reading user profile: {e}")
        return ""

async def update_user_profile(user_id: str, preferences: str):
    """Update user profile preferences."""
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                INSERT INTO user_profiles (user_id, preferences) 
                VALUES (?, ?) 
                ON CONFLICT(user_id) DO UPDATE SET preferences = ?
            """, (user_id, preferences, preferences))
            await db.commit()
            logger.info(f"Updated profile for user {user_id}")
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
