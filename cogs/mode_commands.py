import datetime

import disnake
from disnake.ext import commands

from config import *
from utils.databases import UsersDataBase


class ModeCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = UsersDataBase()

    """Команды для модераторов и админов"""
    @commands.slash_command(description='Kicks the user out of the server', usage='kick <@user> <reason=None>')
    @commands.has_permissions(kick_members=True, administrator=True)
    async def kick(self, interaction, member: disnake.Member, *, reason=None):
        await member.kick(reason=reason)

        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title=f'✅Kick of the user - {member}')
        embed.description = f'{interaction.author.mention} kicked {member.mention}'
        if reason:
            embed.add_field(name='Reason', value=f'{reason}')

        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Bans the user from the server', usage='ban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def ban(self, interaction, user: disnake.User, *, reason=None):
        await interaction.guild.ban(user, reason=reason)
        await self.db.delete_user(interaction.guild.id, user)

        message = f'User {user.mention} was banned'
        if reason:
            message += f'\nReason: {reason}'

        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title=f'✅Ban of the user - {user}')
        embed.description = f'{interaction.author.mention} banned {user.mention}'
        if reason:
            embed.add_field(name='Reason', value=f'{reason}')
        # embed.set_thumbnail(url=member.display_avatar.url)

        if type(interaction) == commands.context.Context:
            await interaction.send(embed=embed)
        elif interaction.response.is_done():
            await interaction.followup.send(embed=embed, ephemeral=HIDDEN_ANSWERS)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Unban the user on the server', usage='unban <@user> <reason=None>')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def unban(self, interaction, user: disnake.User):
        await interaction.guild.unban(user)

        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title=f'✅Unban of the user - {user}')
        embed.description = f'{interaction.author.mention} unbanned {user.mention}'

        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Mutes the user for a certain time')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def mute(self, interaction, member: disnake.Member, minutes: int = commands.Param(gt=0), reason: str = None):
        await member.timeout(reason=reason, duration=datetime.timedelta(minutes=minutes))

        message = f'User {member.mention} is muted for {minutes} minutes'
        if reason:
            message += f'\nReason: {reason}'

        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title=f'✅Mute of the user - {member}')
        embed.description = f'{interaction.author.mention} muted {member.mention} for {minutes} minutes'
        if reason:
            embed.add_field(name='Reason', value=f'{reason}')

        if type(interaction) == commands.context.Context:
            await interaction.send(embed=embed)
        elif interaction.response.is_done():
            await interaction.followup.send(embed=embed, ephemeral=HIDDEN_ANSWERS)
        else:
            await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Unmute the user')
    @commands.has_permissions(mute_members=True, administrator=True)
    async def unmute(self, interaction, member: disnake.Member):
        await member.timeout(reason=None, until=None)

        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title=f'✅Unmute of the user - {member}')
        embed.description = f'{interaction.author.mention} unmuted {member.mention}'

        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Clears the chat from a certain number of messages', usage='clear <count>')
    @commands.has_permissions(manage_messages=True, administrator=True)
    async def clear(self, interaction, msg_count: int = commands.Param(gt=0)):
        """Очистка определенного количества сообщений."""
        embed = disnake.Embed(color=INFO_COLOR,
                              title='🧹Clearing the chat',
                              description=f'🔄️Deleting {msg_count} messages...')
        await interaction.response.send_message(embed=embed, ephemeral=True)

        await interaction.channel.purge(limit=msg_count)
        
        embed = disnake.Embed(color=SUCCESS_COLOR,
                              title='🧹Clearing the chat',
                          description=f'✅Deleted {msg_count} messages!')
        await interaction.edit_original_message(embed=embed)


def setup(bot):
    bot.add_cog(ModeCommands(bot))
