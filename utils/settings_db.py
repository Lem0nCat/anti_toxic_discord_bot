import os

import aiosqlite
import disnake


class SettingsDataBase:
    def __init__(self):
        self.name = 'dbs/settings.db'
        if not os.path.isdir('dbs'):
            os.mkdir('dbs')

    async def dict_factory(cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            d[col[0]] = row[idx]
        return d

    async def create_table(self):
        # Подключение к бд
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'''CREATE TABLE IF NOT EXISTS settings (
                server_id INT PRIMARY KEY,
                bot_msg_delete_time INT DEFAULT 60 NOT NULL,
                user_msg_delete_time INT DEFAULT 60 NOT NULL,
                hidden_answers INT DEFAULT 0 NOT NULL,
                new_user_role_id INT DEFAULT 0 NOT NULL,
                welcome_message INT DEFAULT 1 NOT NULL,
                goodbye_message INT DEFAULT 1 NOT NULL,
                ban_warning_count INT DEFAULT 3 NOT NULL,
                mute_warning_count INT DEFAULT 2 NOT NULL,
                default_mute_duration INT DEFAULT 15 NOT NULL
            )'''
            await cursor.execute(query)
            await db.commit()

    async def server_login(self, server_id: disnake.Guild.id):
        await self.create_table()
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'''INSERT INTO settings (server_id) VALUES (?)'''
            await cursor.execute(query, (server_id, ))
            await db.commit()

    async def get_settings(self, server_id: disnake.Guild.id):
        async with aiosqlite.connect(self.name) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.cursor()
            query = 'SELECT * FROM settings WHERE server_id = ?'
            await cursor.execute(query, (server_id, ))

            result = dict(await cursor.fetchone())
            del result['server_id']
            return result
        
    async def get_one_setting(self, server_id: disnake.Guild.id, setting_name: str):
        await self.create_table()
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'SELECT {setting_name} FROM settings WHERE server_id = ?'
            await cursor.execute(query, (server_id, ))
            result = await cursor.fetchone()
            return result[0]
        
    async def set_settings(self, settings):
        await self.create_table()
        async with aiosqlite.connect(self.name) as db:
            cursor = await db.cursor()
            query = f'''UPDATE settings SET 
                bot_msg_delete_time = ?, 
                user_msg_delete_time = ?, 
                hidden_answers = ?, 
                new_user_role_id = ?, 
                welcome_message = ?, 
                goodbye_message = ?, 
                ban_warning_count = ?, 
                mute_warning_count = ?, 
                default_mute_duration = ?
                WHERE server_id = ?'''
            await cursor.execute(query, settings)
            await db.commit()
