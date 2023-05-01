import datetime

import disnake
from disnake.ext import commands

from config import *


class ModeCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    """Команды для модераторов и админов"""
    @commands.slash_command(description='Kicks the user out of the server', usage='kick <@user> <reason=None>')
    @commands.has_permissions(kick_members=True, administrator=True)
    async def kick(self, interaction, member: disnake.Member, *, reason=None):
        await member.kick(reason=reason)

        message = f'User {member.mention} was kicked'
        if reason:
            message += f'\nReason: {reason}'
        await interaction.response.send_message(message, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Bans the user from the server', usage='ban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def ban(self, interaction, user: disnake.User, *, reason=None):
        await interaction.guild.ban(user, reason=reason)

        message = f'User {user.mention} was banned'
        if reason:
            message += f'\nReason: {reason}'
        await interaction.response.send_message(message, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Unban the user on the server', usage='unban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def unban(self, interaction, user: disnake.User):
        await interaction.guild.unban(user)
        await interaction.response.send_message(f'Unbanned {user.mention}', ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Mutes the user for a certain time')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def mute(self, interaction, member: disnake.Member, minutes: int = commands.Param(gt=0), reason: str = None):
        await member.timeout(reason=reason, duration=datetime.timedelta(minutes=minutes))

        message = f'User {member.mention} is muted for {minutes} minutes'
        if reason:
            message += f'\nReason: {reason}'
        await interaction.response.send_message(message, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Unmute the user')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def unmute(self, interaction, member: disnake.Member):
        await member.timeout(reason=None, until=None)
        await interaction.response.send_message(f'User {member.mention} is unmuted', ephemeral=True)

    @commands.slash_command(description='Clears the chat from a certain number of messages', usage='clear <count>')
    @commands.has_permissions(manage_messages=True, administrator=True)
    async def clear(self, interaction, msg_count: int = commands.Param(gt=0)):
        """Очистка определенного количества сообщений."""
        message = f'Deleting {msg_count} messages...'
        await interaction.response.send_message(message, ephemeral=HIDDEN_ANSWERS)
        # await interaction.response.defer()
        await interaction.channel.purge(limit=msg_count + 1)
        if HIDDEN_ANSWERS:
            await interaction.edit_original_message(f'Deleted {msg_count} messages!')


def setup(bot):
    bot.add_cog(ModeCommands(bot))
