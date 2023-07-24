import disnake
from disnake.ext import commands

from config import *
from utils.users_db import UsersDataBase
from utils.settings_db import SettingsDataBase


class SettingsСommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.users_db = UsersDataBase()
        self.settings_db = SettingsDataBase()

    """Команды для настройки бота для создателя"""
    @commands.slash_command(description='Set up bot settings')
    @commands.has_guild_permissions(administrator=True)
    async def settings(self, interaction, 
                       bot_msg_delete_time: int = None, user_msg_delete_time: int = None, 
                       hidden_answers: bool = None, new_user_role_id: int = None,
                       welcome_message: bool = None, goodbye_message: bool = None,
                       ban_warning_count: int = None, mute_warning_count: int = None, default_mute_duration: int = None):
        try:
            old_settings = await self.settings_db.get_settings(interaction.guild.id)
            settings = [bot_msg_delete_time, user_msg_delete_time, hidden_answers, new_user_role_id, welcome_message, 
                            goodbye_message, ban_warning_count, mute_warning_count, default_mute_duration]
            
            for index, (old, new) in enumerate(zip(old_settings.values(), settings)):
                settings[index] = old if new == None else new

            settings.append(interaction.guild.id)

            await self.settings_db.set_settings(settings)

            embed = disnake.Embed(color=SUCCESS_COLOR,
                                  title=f'✅ You have successfully changed the bot settings')
        except Exception as e:
            print(e)
            embed = disnake.Embed(color=ERROR_COLOR,
                                  title=f'❌ Error when trying to change settings')
        await interaction.response.send_message(embed=embed, ephemeral=True)
            
    @commands.slash_command(description='Shows bot settings on your server')
    @commands.has_guild_permissions(administrator=True)
    async def show_settings(self, interaction):
        try:
            settings = await self.settings_db.get_settings(interaction.guild.id)

            embed = disnake.Embed(color=INFO_COLOR,
                              title='Bot settings on your server')
            
            for key, value in settings.items():
                embed.add_field(name=key, value=f'```{value}```')
        except Exception as e:
            print(e)
            embed = disnake.Embed(color=ERROR_COLOR,
                                  title=f'❌ Error when trying to change settings')
        await interaction.response.send_message(embed=embed, ephemeral=True)


def setup(bot):
    bot.add_cog(SettingsСommands(bot))
