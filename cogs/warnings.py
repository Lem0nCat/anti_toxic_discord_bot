import disnake
from disnake.ext import commands

from config import *
from utils.users_db import UsersDataBase
from utils.settings_db import SettingsDataBase


class Warnings(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.users_db = UsersDataBase()
        self.settings_db = SettingsDataBase()

    @commands.slash_command(description='View the number of your warnings')
    async def warnings(self, interaction, member: disnake.Member = None):
        await self.users_db.create_table(interaction.guild.id)
        if not member:
            member = interaction.author
        user_warnings = await self.users_db.get_user_alerts(interaction.guild.id, member)
        embed = disnake.Embed(color=INFO_COLOR,
                              title=f'Number of user warnings - {member}')
        embed.add_field(name='Current quantity', value=f'```{len(user_warnings)}```')
        embed.add_field(name='Maximum warnings',
                        value=f'```{await self.settings_db.get_one_setting(interaction.guild.id, "ban_warning_count")}```')
        
        for count, value in enumerate(user_warnings):
            embed.add_field(name=f"Warning #{count + 1} | Warning_ID:{value[0]}", value=f'**Reason**: ```{value[2]}```', inline=False)

        embed.set_thumbnail(url=member.display_avatar.url)
        await interaction.response.send_message(embed=embed)

    @commands.slash_command(description='Issue a warning to the user')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def warn(self, interaction, user: disnake.User, reason: str = None):
        await self.users_db.create_table(interaction.guild.id)
        await self.users_db.issue_warnings(interaction.guild.id, user, reason=reason)

        if type(interaction) == commands.context.Context:
            interaction.author = self.bot.user

        embed = disnake.Embed(color=WARNING_COLOR,
                              title=f'⚠️Issuing a warning to the user - {user}')
        embed.description = f'{interaction.author.mention} issued a warning to {user.mention}'
        if reason:
            embed.add_field(name='Reason', value=f'{reason}')

        if type(interaction) == commands.context.Context:
            await interaction.send(embed=embed)
        else:
            await interaction.response.send_message(embed=embed, 
                                                    ephemeral=await self.settings_db.get_one_setting(interaction.guild.id, "hidden_answers"))

        user_warnings = await self.users_db.get_user_alerts_count(interaction.guild.id, user)

        if user_warnings >= await self.settings_db.get_one_setting(interaction.guild.id, "ban_warning_count"):
            await self.bot.get_slash_command('ban').callback(self, interaction=interaction, user=user)
        elif user_warnings >= await self.settings_db.get_one_setting(interaction.guild.id, "mute_warning_count"):
            member = await interaction.guild.fetch_member(user.id)
            if member:
                await self.bot.get_slash_command('mute').callback(self, interaction=interaction, 
                                                                  member=user, 
                                                                  minutes=await self.settings_db.get_one_setting(interaction.guild.id, "default_mute_duration"), 
                                                                  reason='Numerous violations of the rules')

    @commands.slash_command(description='Removes one warning from the user')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def unwarn(self, interaction, warning_id: int):
        try:
            user_id = await self.users_db.delete_warning(interaction.guild.id, warning_id=warning_id)
            user = await self.bot.fetch_user(user_id)

            embed = disnake.Embed(color=WARNING_COLOR,
                                  title=f'⚠️Removing warning')
            embed.description = f'{interaction.author.mention} removed the warning from {user.mention}'
        except:
            embed = disnake.Embed(color=ERROR_COLOR,
                                  title=f'❌Removing warning')
            embed.description = f'Warning with given id was not found'

        await interaction.response.send_message(embed=embed, 
                                                ephemeral=await self.settings_db.get_one_setting(interaction.guild.id, "hidden_answers"))

    @commands.slash_command(description='Removes all warnings from one user')
    @commands.has_permissions(ban_members=True, administrator=True)
    async def сlear_user_warnings(self, interaction, user: disnake.User):
        user_warnings = await self.users_db.get_user_alerts_count(interaction.guild.id, user)
        if user_warnings > 0:
            await self.users_db.delete_all_warnings(interaction.guild.id, user)
            embed = disnake.Embed(color=WARNING_COLOR,
                                  title=f'⚠️Removes all warnings from the user - {user}')
            embed.description = f'{interaction.author.mention} removed all warnings from {user.mention}'
        else:
            embed = disnake.Embed(color=ERROR_COLOR,
                                  title=f'❌Removes all warnings from the user - {user}')
            embed.description = f'This user has no warnings!'

        await interaction.response.send_message(embed=embed, 
                                                ephemeral=await self.settings_db.get_one_setting(interaction.guild.id, "hidden_answers"))


def setup(bot):
    bot.add_cog(Warnings(bot))
