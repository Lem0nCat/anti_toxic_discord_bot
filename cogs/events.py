import disnake
from disnake.ext import commands

from config import *


class Events(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    """Ивенты"""
    @commands.Cog.listener()
    async def on_ready(self):
        print(f'The bot {self.bot.user} is ready to work!')

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        print(error)

        if isinstance(error, commands.MissingPermissions):
            await ctx.reply(embed=disnake.Embed(
                description=f"**❌ {ctx.author.name}, you don't have enough permissions to use this command!**\n",
                color=ERROR_COLOR
            ), delete_after=BOT_MSG_TIME_DELETE)
            await ctx.message.delete(delay=USER_MSG_TIME_DELETE)

        elif isinstance(error, commands.UserInputError):
            await ctx.reply(embed=disnake.Embed(
                description=f'**❌ {ctx.author.name}, the command was entered incorrectly!**\n'
                            f'Use: `{ctx.prefix}{ctx.command.name}` ({ctx.command.brief})\n'
                            f'Example: `{ctx.prefix}{ctx.command.usage}`',
                color=ERROR_COLOR
            ), delete_after=BOT_MSG_TIME_DELETE)
            await ctx.message.delete(delay=USER_MSG_TIME_DELETE)

    @commands.Cog.listener()
    async def on_slash_command_error(self, interaction, error: Exception) -> None:
        print(error)

        if isinstance(error, commands.MissingPermissions):
            await interaction.response.send_message(embed=disnake.Embed(
                description=f"**❌ {interaction.author.name}, you don't have enough permissions to use this command!**\n",
                color=ERROR_COLOR
            ), delete_after=BOT_MSG_TIME_DELETE)

    @commands.Cog.listener()
    async def on_member_join(self, member):
        if not WELCOME_MESSAGE:
            return

        channel = member.guild.system_channel

        embed = disnake.Embed(
            title="**Welcome**",
            description=f'{member.mention} has joined our community',
            color=INFO_COLOR
        )
        embed.set_thumbnail(url=member.avatar.url)

        if NEW_USER_ROLE_ID:
            role = disnake.utils.get(member.guild.roles, id=NEW_USER_ROLE_ID)
            await member.add_roles(role)

        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_member_remove(self, member):
        if not GOODBYE_MESSAGE:
            return

        channel = member.guild.system_channel

        embed = disnake.Embed(
            title="**Goodbye**",
            description=f'{member.mention} has left our community...',
            color=INFO_COLOR
        )
        embed.set_thumbnail(url=member.avatar.url)

        await channel.send(embed=embed)


def setup(bot):
    bot.add_cog(Events(bot))
