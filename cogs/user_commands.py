import disnake
from disnake import ApplicationCommandInteraction
from disnake.ext import commands


class UserCommands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    """Обычные команды"""
    @commands.command(pass_context=True)
    async def ping(self, ctx):
        await ctx.send('Pong!')

    @commands.slash_command()
    async def confirm(self, inter: ApplicationCommandInteraction):
        await inter.response.send_modal(
            title="Подтверждение",
            custom_id="confirm-or-deny",
            components=[disnake.ui.TextInput(label="подтвердить?", custom_id="confirm")],
        )
        await inter.followup.send(content="Пожалуйста, не закрывайте модальное окно!", ephemeral=True)


def setup(bot):
    bot.add_cog(UserCommands(bot))