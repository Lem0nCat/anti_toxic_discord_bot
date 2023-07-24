import disnake
from disnake.ext import commands

from config import *
from utils.settings_db import SettingsDataBase


class Events(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.settings_db = SettingsDataBase()

    """Ивенты"""
    @commands.Cog.listener()
    async def on_ready(self):
        print(f'The bot {self.bot.user} is ready to work!')

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        await self.settings_db.server_login(guild.id)

    @commands.Cog.listener()
    async def on_member_join(self, member):
        if not await self.settings_db.get_one_setting(member.guild.id, "welcome_message"):
            return

        channel = member.guild.system_channel

        embed = disnake.Embed(
            title="**Welcome**",
            description=f'{member.mention} has joined our community',
            color=INFO_COLOR
        )
        embed.set_thumbnail(url=member.avatar.url)

        if await self.settings_db.get_one_setting(member.guild.id, "new_user_role_id"):
            role = disnake.utils.get(member.guild.roles, id=await self.settings_db.get_one_setting(member.guild.id, "new_user_role_id"))
            if role:
                await member.add_roles(role)

        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_member_remove(self, member):
        if not await self.settings_db.get_one_setting(member.guild.id, "goodbye_message"):
            return

        channel = member.guild.system_channel

        embed = disnake.Embed(
            title="**Goodbye**",
            description=f'{member.mention} has left our community...',
            color=INFO_COLOR
        )
        embed.set_thumbnail(url=member.avatar.url)

        await channel.send(embed=embed)

    @commands.Cog.listener()
    async def on_command_error(self, ctx, error):
        print(error)

        if isinstance(error, commands.MissingPermissions):
            await ctx.reply(embed=disnake.Embed(
                description=f"**❌ {ctx.author.name}, you don't have enough permissions to use this command!**\n",
                color=ERROR_COLOR
            ), delete_after=await self.settings_db.get_one_setting(ctx.guild.id, "bot_msg_delete_time"))
            await ctx.message.delete(delay=await self.settings_db.get_one_setting(ctx.guild.id, "user_msg_delete_time"))

        elif isinstance(error, commands.UserInputError):
            await ctx.reply(embed=disnake.Embed(
                description=f'**❌ {ctx.author.name}, the command was entered incorrectly!**\n'
                            f'Use: `{ctx.prefix}{ctx.command.name}` ({ctx.command.brief})\n'
                            f'Example: `{ctx.prefix}{ctx.command.usage}`',
                color=ERROR_COLOR
            ), delete_after=await self.settings_db.get_one_setting(ctx.guild.id, "bot_msg_delete_time"))
            await ctx.message.delete(delay=await self.settings_db.get_one_setting(ctx.guild.id, "user_msg_delete_time"))

    @commands.Cog.listener()
    async def on_slash_command_error(self, interaction, error: Exception) -> None:
        print(error)

        if isinstance(error, commands.MissingPermissions):
            await interaction.response.send_message(embed=disnake.Embed(
                description=f"**❌ {interaction.author.name}, you don't have enough permissions to use this command!**\n",
                color=ERROR_COLOR
            ), delete_after=await self.settings_db.get_one_setting(interaction.guild.id, "bot_msg_delete_time"))


def setup(bot):
    bot.add_cog(Events(bot))
