import disnake
from disnake.ext import commands

from config import *
from utils.databases import UsersDataBase


class Warnings(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.db = UsersDataBase()

    @commands.slash_command(description='View the number of your warnings')
    async def warnings(self, interaction, member: disnake.Member = None):
        await self.db.create_table(interaction.guild.id)
        if not member:
            member = interaction.author
        await self.db.add_user(interaction.guild.id, member)
        user = await self.db.get_user(interaction.guild.id, member)
        embed = disnake.Embed(color=INFO_COLOR,
                              title=f'Number of user warnings - {member}')
        embed.add_field(name='Current quantity', value=f'```{user[1]}```')
        embed.add_field(name='Maximum warnings', value=f'```{MAX_WARNINGS}```')
        embed.set_thumbnail(url=member.display_avatar.url)
        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

    @commands.slash_command(description='Issue a warning to the user')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def warn(self, interaction, user: disnake.User, reason: str = None):
        await self.db.create_table(interaction.guild.id)
        await self.db.add_user(interaction.guild.id, user)
        await self.db.issue_warnings(interaction.guild.id, user, 1)

        embed = disnake.Embed(color=WARNING_COLOR,
                              title=f'⚠️Issuing a warning to the user - {user}')
        embed.description = f'{interaction.author.mention} issued a warning to {user.mention}'
        if reason:
            embed.add_field(name='Reason', value=f'{reason}')
        # embed.set_thumbnail(url=member.display_avatar.url)
        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)

        user_bd = await self.db.get_user(interaction.guild.id, user)
        if user_bd[1] >= MAX_WARNINGS:
            await self.bot.get_slash_command("ban").callback(self, interaction=interaction, user=user)

    @commands.slash_command(description='Removes one warning from the user')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def unwarn(self, interaction, user: disnake.User):
        user_bd = await self.db.get_user(interaction.guild.id, user)
        if user_bd and user_bd[1] > 0:
            await self.db.issue_warnings(interaction.guild.id, user, -1)
            embed = disnake.Embed(color=WARNING_COLOR,
                                  title=f'⚠️Removes one warning from the user - {user}')
            embed.description = f'{interaction.author.mention} withdrew the warning from {user.mention}'
        else:
            embed = disnake.Embed(color=ERROR_COLOR,
                                  title=f'❌Removes one warning from the user - {user}')
            embed.description = f'This user has no warnings!'

        await interaction.response.send_message(embed=embed, ephemeral=HIDDEN_ANSWERS)


def setup(bot):
    bot.add_cog(Warnings(bot))
