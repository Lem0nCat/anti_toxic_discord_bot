# Anti-Toxic Discord Bot

## Description
The Anti-Toxic Discord Bot is a Discord bot that helps maintain a friendly atmosphere on your server by automatically monitoring and removing toxic messages. Using machine learning and natural language processing, the bot identifies toxic messages and takes appropriate actions.

## Key Features
- **Toxic Message Detection**: Uses a machine learning model to identify toxic comments.
- **Message Deletion**: Automatically deletes toxic messages.
- **Warnings**: Sends warnings to users who send toxic messages.
- **Logging**: Keeps a log of all bot actions.

## Requirements
- Python 3.8+
- Discord.py
- PyTorch
- Pandas
- Scikit-learn

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Lem0nCat/anti_toxic_discord_bot.git
    cd anti_toxic_discord_bot
    ```

2. **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3. **Create and configure the configuration file:**
    - Locate the `config.py` file in the root directory of the project.
    - Insert your Discord bot token:
      ```py
      BOT_TOKEN = 'your_bot_token'
      ```
    - Modify other parameters as necessary.

4. **Run the bot:**
    ```bash
    python bot.py
    ```

## Usage
- Once the bot is added to your server, it will automatically start monitoring messages.
- The bot will delete messages it classifies as toxic and send warnings to users.
- All bot actions will be logged for later analysis.

## Configuration
In the `config.py` file, you can configure the following parameters:
- **Prefix**: The prefix for bot commands.
- **Colors**: Notification colors used by the bot.
- **Models**: Neural network models for detecting toxicity in English and Russian text.

## Model Development and Training
- **Model Training**: Scripts for training the neural network model can be found in the `nn` folder.
- **Data**: Use datasets containing toxic and non-toxic messages to train the model.
- **Testing**: Scripts for testing the accuracy of the model are located in the `tests` folder.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
