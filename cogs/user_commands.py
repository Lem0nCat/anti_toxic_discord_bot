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

def setup(bot):
    bot.add_cog(UserCommands(bot))