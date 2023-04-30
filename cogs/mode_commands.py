import datetime

import disnake
from disnake.ext import commands

from config import *


class ModeCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    """Команды для модераторов и админов"""
    @commands.command(description='Kicks the user out of the server', usage='kick <@user> <reason=None>')
    @commands.has_permissions(kick_members=True, administrator=True)
    async def kick(self, ctx, member: disnake.Member, *, reason='Нарушение правил.'):
        await ctx.send(f'Администратор {ctx.author.mention} исключил пользователя {member.mention}',
                       delete_after=BOT_MSG_TIME_DELETE)
        await member.kick(reason=reason)
        await ctx.message.delete(delay=USER_MSG_TIME_DELETE)

    @commands.slash_command(description='Bans the user from the server', usage='ban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def ban(self, interaction, user: disnake.User, *, reason=None):
        await interaction.guild.ban(user, reason=reason)
        await interaction.response.send_message(f'Banned {user.mention}', ephemeral=True)

        # await ctx.send(f'Администратор {ctx.author.mention} забанил пользователя {member.mention}',
        #                delete_after=BOT_MSG_TIME_DELETE)
        # await member.ban(reason=reason)
        # await ctx.message.delete(delay=USER_MSG_TIME_DELETE)

    @commands.slash_command(description='Unban the user on the server', usage='unban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def unban(self, interaction, user: disnake.User):
        await interaction.guild.unban(user)
        await interaction.response.send_message(f'Unbanned {user.mention}', ephemeral=True)

        # await ctx.send(f'Администратор {ctx.author.mention} разбанил пользователя {member.mention}',
        #                delete_after=BOT_MSG_TIME_DELETE)
        # await member.unban()
        # await ctx.message.delete(delay=USER_MSG_TIME_DELETE)

    @commands.slash_command(description='Mutes the user for a certain time')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def mute(self, interaction, member: disnake.Member, minutes: int = commands.Param(gt=0), reason: str = None):

        await member.timeout(reason=reason, duration=datetime.timedelta(minutes=minutes))

        message = f'User {member.mention} is muted for {minutes} minutes'
        if reason:
            message += f'\nReason: {reason}'
        await interaction.response.send_message(message, ephemeral=True)

    @commands.slash_command(description='Unmute the user')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def unmute(self, interaction, member: disnake.Member):
        await member.timeout(reason=None, until=None)
        await interaction.response.send_message(f'User {member.mention} is unmuted', ephemeral=True)

    @commands.command(brief='Удаляет определенное количество сообщений в чате', usage='clear <count>')
    @commands.has_permissions(manage_messages=True, administrator=True)
    async def clear(self, ctx, msg_count=0):
        """Очистка определенного количества сообщений."""
        if msg_count > 0:
            await ctx.channel.purge(limit=msg_count + 1)
        else:
            bot_msg = await ctx.reply(embed=disnake.Embed(
                description=f'**❌ {ctx.author.name}, the command was entered incorrectly!**\n'
                            'To clear the chat, use - `!clear <count>`',
                color=ERROR_COLOR
            ))
            await bot_msg.delete(delay=BOT_MSG_TIME_DELETE)
            await ctx.message.delete(delay=USER_MSG_TIME_DELETE)


def setup(bot):
    bot.add_cog(ModeCommands(bot))
