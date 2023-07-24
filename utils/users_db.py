import os

import aiosqlite
import disnake

from config import *


class UsersDataBase:
    def __init__(self):
        self.name = 'dbs/users.db'
        if not os.path.isdir('dbs'):
            os.mkdir('dbs')

    async def create_table(self, server_id: disnake.Guild.id):
        # Подключение к бд
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'''CREATE TABLE IF NOT EXISTS s{server_id} (  
                id INTEGER PRIMARY KEY,
                user_id INTEGER,
                reason VARCHAR(100)
            )'''
            await cursor.execute(query)    # Выполнение запроса
            await db.commit()   # Сохранение изменений

    async def get_user_alerts_count(self, server_id: disnake.Guild.id, user: disnake.Member):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'SELECT COUNT(*) FROM s{server_id} WHERE user_id = ?'
            await cursor.execute(query, (user.id, ))
            count = await cursor.fetchone()
            return count[0]
        
    async def get_user_alerts(self, server_id: disnake.Guild.id, user: disnake.Member):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'SELECT * FROM s{server_id} WHERE user_id = ?'
            await cursor.execute(query, (user.id, ))
            return await cursor.fetchall()

    async def delete_warning(self, server_id: disnake.Guild.id, warning_id: int):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'SELECT user_id FROM s{server_id} WHERE id = ?'
            await cursor.execute(query, (warning_id, ))
            user_id = await cursor.fetchone()

            query = f'DELETE FROM s{server_id} WHERE id = ?'
            await cursor.execute(query, (warning_id, ))
            await db.commit()

            return user_id[0]

    async def delete_all_warnings(self, server_id: disnake.Guild.id, user: disnake.User):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'DELETE FROM s{server_id} WHERE user_id = ?'
            await cursor.execute(query, (user.id, ))
            await db.commit()

    async def issue_warnings(self, server_id: disnake.Guild.id, user: disnake.Member, reason: str):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'INSERT INTO s{server_id} (user_id, reason) VALUES (?, ?)'
            await cursor.execute(query, (user.id, reason))
            await db.commit()

