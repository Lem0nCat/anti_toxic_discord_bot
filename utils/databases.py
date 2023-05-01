import aiosqlite
import disnake

from config import *


class UsersDataBase:
    def __init__(self):
        self.name = 'dbs/users.db'

    async def create_table(self, server_id: disnake.Guild.id):
        # Подключение к бд
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'''CREATE TABLE IF NOT EXISTS s{server_id} (  
                id INTEGER PRIMARY KEY,
                warnings_number INTEGER
            )'''
            await cursor.execute(query)    # Выполнение запроса
            await db.commit()   # Сохранение изменений

    async def get_user(self, server_id: disnake.Guild.id, user: disnake.Member):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'SELECT * FROM s{server_id} WHERE id = ?'
            await cursor.execute(query, (user.id, ))
            return await cursor.fetchone()

    # async def get_warnings(self, server_id: disnake.Guild.id, user: disnake.Member):
    #     bd_user = self.get_user(server_id, user)
    #     if bd_user:
    #         return bd_user[1]
    #     else:
    #         return None
    
    async def add_user(self, server_id: disnake.Guild.id, user: disnake.Member):
        async with aiosqlite.connect(self.name) as db:
            if not await self.get_user(server_id, user):
                cursor = await db.cursor()
                query = f'INSERT INTO s{server_id} (id, warnings_number) VALUES (?, ?)'
                await cursor.execute(query, (user.id, 0))
                await db.commit()

    async def delete_user(self, server_id: disnake.Guild.id, user: disnake.Member):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'DELETE FROM s{server_id} WHERE id = ?'
            await cursor.execute(query, (user.id, ))
            await db.commit()

    async def issue_warnings(self, server_id: disnake.Guild.id, user: disnake.Member, count: int):
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'UPDATE s{server_id} SET warnings_number = warnings_number + ? WHERE id = ?'
            await cursor.execute(query, (count, user.id))
            await db.commit()
