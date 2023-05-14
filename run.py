import os
import disnake
from disnake.ext import commands

from config import BOT_TOKEN, PREFIX


bot = commands.Bot(command_prefix=PREFIX, help_command=None, intents=disnake.Intents.all(), test_guilds=[927622577218793502])


@bot.command(brief='Load a certain cog', usage='load <cog_name>')
@commands.is_owner()
async def load(ctx, extension):
    bot.load_extension(f'cogs.{extension}')
    print(f'Loaded extension "cogs.{extension}"')


@bot.command(brief='Unload a certain cog', usage='unload <cog_name>')
@commands.is_owner()
async def unload(ctx, extension):
    bot.unload_extension(f'cogs.{extension}')
    print(f'Unloaded extension "cogs.{extension}"')


@bot.command(brief='Reload a certain cog', usage='reload <cog_name>')
@commands.is_owner()
async def reload(ctx, extension):
    bot.reload_extension(f'cogs.{extension}')
    print(f'Reloaded extension "cogs.{extension}"')


for filename in os.listdir('cogs'):
    if filename.endswith(".py"):
        bot.load_extension(f'cogs.{filename[:-3]}')
        print(f'Loaded extension "cogs.{filename[:-3]}"')


bot.run(BOT_TOKEN)
