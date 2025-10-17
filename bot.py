import logging
import re
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler

def get_action_keyboard(actions, help_button=False):
    """Create a keyboard with action buttons and optional help button
    
    Args:
        actions (list): List of action button labels
        help_button (bool): Whether to add a help button
    """
    # Create action buttons with consistent casing - first letter capitalized
    keyboard = [[KeyboardButton(action.capitalize())] for action in actions]
    
    # Add help button in a separate row if requested
    if help_button:
        keyboard.append([KeyboardButton("❓ Help")])
        
    return ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
import os
from dotenv import load_dotenv
import pyreadstat
import tempfile
# Lazy import: weights_handler импортируется при обработке файла
from spss_handlers import validate_spss_file, read_spss_with_fallbacks, SPSSReadError
import atexit, sys

LOCK_FILE = os.path.join(tempfile.gettempdir(), "tg_bot.lock")

def acquire_lock():
    try:
        fd = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        os.write(fd, str(os.getpid()).encode())
        atexit.register(lambda: os.path.exists(LOCK_FILE) and os.remove(LOCK_FILE))
    except FileExistsError:
        print("Bot already running. Exit.", file=sys.stderr)
        sys.exit(1)

# Load environment variables
load_dotenv()

# Настройка логирования
from logging.handlers import RotatingFileHandler

# Configure logging to write debug messages to Telegram
async def send_debug_message(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Send a debug message to the bot owner"""
    try:
        # Replace with your Telegram user ID
        OWNER_ID = None  # We'll get this from the first user to use /start
        if OWNER_ID and context and getattr(context, 'bot', None):
            await context.bot.send_message(OWNER_ID, f"DEBUG: {message}")
    except Exception as e:
        pass  # Silently fail if we can't send debug messages

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
# 1. File handler with rotation
file_handler = RotatingFileHandler(
    'bot.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)

# 2. Telegram message handler for in-memory logs
class TelegramMessageHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.messages.append(msg)
            # Keep only last 100 messages
            if len(self.messages) > 100:
                self.messages.pop(0)
        except Exception:
            self.handleError(record)

telegram_handler = TelegramMessageHandler()
telegram_handler.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter(
    '%(asctime)s [PID:%(process)d] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S.%f'
)
telegram_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set formatters
file_handler.setFormatter(file_formatter)
telegram_handler.setFormatter(telegram_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(telegram_handler)

# Ensure all messages are captured
logger.propagate = False

# Add a command to view logs
async def view_logs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the last 10 log messages to the user"""
    if not telegram_handler.messages:
        await update.message.reply_text("No logs available.")
        return
    
    log_text = "\n".join(telegram_handler.messages[-10:])  # Last 10 messages
    await update.message.reply_text(f"Last 10 log messages:\n\n{log_text}")

# Log startup
logger.info("Bot starting up...")
logger.info("Logging configured to store messages in memory")

# Состояния диалога
(UPLOAD_FILE, SELECT_DEPENDENT, SELECT_INDEPENDENT, 
 SELECT_SUBGROUPS, CONFIRM_ANALYSIS) = range(5)

def parse_variable_list(text: str) -> list:
    """Parse a comma-separated list of variables, handling trailing commas and whitespace
    
    Args:
        text (str): Comma-separated list of variables
        
    Returns:
        list: List of cleaned variable names
    """
    # First clean up the text
    cleaned_text = text.strip()
    # Remove multiple consecutive commas
    while ',,' in cleaned_text:
        cleaned_text = cleaned_text.replace(',,', ',')
    # Remove trailing comma if present
    if cleaned_text.endswith(','):
        cleaned_text = cleaned_text[:-1]
    
    # If we have nothing left after cleanup, return empty list
    if not cleaned_text:
        return []
    
    # Split by comma and clean each variable name
    variables = [var.strip(" '\"""''") for var in cleaned_text.split(',')]
    # Remove any remaining empty strings
    variables = [var for var in variables if var]
    return variables

def match_variables_case_insensitive(user_vars: list, available_vars: list) -> tuple:
    """Match user-provided variable names to available variables (case-insensitive)
    
    Args:
        user_vars (list): Variable names provided by user
        available_vars (list): Available variable names in the dataset
        
    Returns:
        tuple: (matched_vars, invalid_vars)
            matched_vars: list of correctly-cased variable names from the dataset
            invalid_vars: list of user variables that couldn't be matched
    """
    # Create a lowercase mapping for case-insensitive lookup
    lowercase_mapping = {var.lower(): var for var in available_vars}
    
    matched_vars = []
    invalid_vars = []
    
    for user_var in user_vars:
        user_var_lower = user_var.lower()
        if user_var_lower in lowercase_mapping:
            # Use the correctly-cased variable name from the dataset
            matched_vars.append(lowercase_mapping[user_var_lower])
        else:
            # Variable not found even with case-insensitive match
            invalid_vars.append(user_var)
    
    return matched_vars, invalid_vars

def format_variable_list_message(variables: list, var_type: str) -> str:
    """Format variable list message, handling long lists
    
    Args:
        variables: List of variable names
        var_type: Type of variables ('dependent', 'independent', 'subgroups')
    
    Returns:
        str: Formatted message
    """
    # Constants for display
    MAX_MESSAGE_LENGTH = 3000  # Leave buffer for Telegram's 4096 limit and Markdown
    
    # Headers for different variable types
    headers = {
        'dependent': "🎯 *Step 1: Select **Dependent Variables***",
        'independent': "📊 *Step 2: Select **Independent Variables***",
        'subgroups': "🔍 *Step 3: Select **Subgroup Variables** (Optional)*"
    }
    
    # Instructions
    instructions = {
        'dependent': "Type variable names separated by commas:",
        'independent': "Type variable names separated by commas (min 2):",
        'subgroups': "Type variable names or press Skip:"
    }
    
    # Build the basic message
    header = headers.get(var_type, "Select Variables")
    instruction = instructions.get(var_type, "Type variable names:")
    
    # Calculate if we need to truncate
    full_list = ', '.join(variables)
    base_message = f"{header}\n━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n{instruction}\n"
    help_text = "\n\nPress ❓ Help for more information"
    
    # Check if the full message would be too long
    test_message = f"{base_message}```\n{full_list}\n```{help_text}"
    
    if len(test_message) <= MAX_MESSAGE_LENGTH:
        # Small enough - send normally
        return test_message
    else:
        # Too long - just ask user to type variable names
        message = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚠️ **В файле найдено {len(variables)} переменных**\n"
            f"⚠️ **Found {len(variables)} variables in the file**\n\n"
            f"Список слишком большой для отображения.\n"
            f"The list is too large to display.\n\n"
            f"✏️ **Пожалуйста, впишите названия нужных вам переменных через запятую**\n"
            f"✏️ **Please type the names of the variables you need, separated by commas**\n\n"
            f"Пример / Example:\n"
            f"```\nvar1, var2, var3, var4\n```\n"
            f"{help_text}"
        )
        return message

# Help messages for each variable type
VARIABLE_HELP = {
    'dependent': (
        "*About Dependent Variables:*\n"
        "• Your target metrics to analyze\n"
        "• The outcome variables you want to understand\n"
        "• Example: customer satisfaction score\n"
        "• Can select multiple variables\n\n"
        "Type variable names or press Cancel to start over."
    ),
    'independent': (
        "*About Independent Variables:*\n"
        "• Variables that might influence your dependent variables\n"
        "• These are your predictor variables\n"
        "• Example: service quality, price satisfaction\n"
        "• Must select at least 2 variables\n\n"
        "Type variable names or press Cancel to start over."
    ),
    'subgroups': (
        "*About Subgroup Variables:*\n"
        "• Variables to segment your analysis by\n"
        "• Brand variables can be included as subgroups\n"
        "• Analysis will be performed for each value\n"
        "• Example: region, customer type, brand\n\n"
        "Type variable names, press Skip, or Cancel to start over."
    )
}

# Хранилище данных пользователей
user_data = {}

def get_user_data(user_id):
    """Safely get user data"""
    if user_id not in user_data:
        return None
    return user_data[user_id].copy()

def update_user_data(user_id, new_data):
    """Safely update user data"""
    user_data[user_id] = new_data

def cleanup_user_data(user_id, delete_file=True):
    """Clean up user data while preserving essential information"""
    if user_id in user_data:
        if delete_file and "file_path" in user_data[user_id]:
            try:
                os.unlink(user_data[user_id]["file_path"])
            except:
                pass
        # Keep essential data for variable selection
        user_data[user_id] = {
            k: v for k, v in user_data[user_id].items() 
            if k in ['numeric_vars', 'all_vars', 'categorical_vars', 'state', 'meta', 'n_rows', 'n_cols']
        }

# Get token from environment variable
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    raise ValueError("Telegram bot token not found in environment variables. Please set TELEGRAM_BOT_TOKEN in .env file.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handler for /start command"""
    user = update.effective_user
    welcome_message = (
        f"👋 Hello {user.first_name}! I'm the Johnson's Relative Weights Calculator Bot.\n\n"
        "I can help you calculate Johnson's Relative Weights from your SPSS data files.\n\n"
        "To get started:\n"
        "1. Make sure your data is in SPSS format (.sav)\n"
        "2. Send me your .sav file\n"
        "3. I'll process it and return the results\n\n"
        "Use /help to see all available commands."
    )
    await update.message.reply_text(welcome_message)
    return UPLOAD_FILE

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /help command"""
    user_id = update.effective_user.id
    
    # Get current state info
    current_state = "Not in a session"
    available_vars = "None"
    selected_vars = []
    
    if user_id in user_data:
        current_state = user_data[user_id].get('state', 'Unknown')
        
        # Get available variables if any
        if 'numeric_vars' in user_data[user_id]:
            available_vars = ", ".join(user_data[user_id]['numeric_vars'])
        
        # Get selected variables if any
        if 'dependent_vars' in user_data[user_id]:
            selected_vars.append(f"Dependent: {', '.join(user_data[user_id]['dependent_vars'])}")
        if 'independent_vars' in user_data[user_id]:
            selected_vars.append(f"Independent: {', '.join(user_data[user_id]['independent_vars'])}")
        if 'subgroups' in user_data[user_id]:
            selected_vars.append(f"Subgroups: {', '.join(user_data[user_id]['subgroups'])}")
    
    help_message = (
        "🤖 *Калькулятор относительных весов Джонсона*\n\n"
        "*О боте:*\n"
        "Этот бот вычисляет относительные веса Джонсона из ваших SPSS-файлов.\n\n"
        "*Инструкции:*\n"
        "1. Используйте /start для начала\n"
        "2. Загрузите ваш файл в формате .sav\n"
        "3. Дождитесь результатов расчета\n\n"
        "*Требования к файлам:*\n"
        "- Формат SPSS (.sav)\n"
        "- Максимальный размер файла: 50MB\n\n"
        "*Поддерживаемые функции:*\n"
        "- Различные кодировки файлов (UTF-8, CP1251, Latin1)\n"
        "- Файлы с пробелами и спецсимволами в именах\n"
        "- Автоматическая фильтрация метаданных\n"
        "- Многоязычные имена и метки переменных\n"
        "- Обработка пропущенных значений (missing values, коды 98 и 99)\n\n"
        "*Обработка данных:*\n"
        "- Автоматически исключает колонки метаданных\n"
        "- Фильтрует переменные с нулевой дисперсией\n"
        "- Использует множественную импутацию для обработки пропущенных значений\n"
        "  (итеративная импутация с использованием ExtraTreesRegressor для\n"
        "  предсказания пропущенных значений на основе других переменных)\n"
        "  Подробное описание подходов к импутации: Multiple Imputations Readme.txt\n\n"
        "*Доступные команды:*\n"
        "/start - Начать новый расчет\n"
        "/help - Показать это сообщение\n"
        "/cancel - Отменить текущую операцию\n\n"
        "*Нужна помощь?*\n"
        "Для технической поддержки обратитесь к @BaukoshaWorks\n\n"
        "🤖 *Johnson's Relative Weights Calculator Bot Help*\n\n"
        "*About:*\n"
        "This bot calculates Johnson's Relative Weights from your SPSS data files.\n\n"
        "*Instructions:*\n"
        "1. Use /start to begin\n"
        "2. Upload your .sav format data file\n"
        "3. Wait for the calculation results\n\n"
        "*File Requirements:*\n"
        "- SPSS format (.sav)\n"
        "- Maximum file size: 50MB\n\n"
        "*Supported Features:*\n"
        "- Multiple file encodings (UTF-8, CP1251, Latin1)\n"
        "- Files with spaces and special characters in names\n"
        "- Automatic metadata column filtering\n"
        "- Multi-language variable names and labels\n"
        "- Missing values handling\n\n"
        "*Data Processing:*\n"
        "- Automatically excludes metadata columns\n"
        "- Filters out zero-variance variables\n"
        "- Uses multiple imputation for missing values\n"
        "  (iterative imputation using ExtraTreesRegressor to\n"
        "  predict missing values based on other variables)\n"
        "  For detailed description see: Multiple Imputations Readme.txt\n\n"
        "*Available Commands:*\n"
        "/start - Start a new calculation\n"
        "/help - Show this help message\n"
        "/cancel - Cancel current operation\n\n"
        "*Need More Help?*\n"
        "For technical support, please contact @BaukoshaWorks"
    )
    await update.message.reply_text(help_message, parse_mode='Markdown')

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /debug command - shows current state and data"""
    user_id = update.effective_user.id
    
    # Get current state info
    current_state = "Unknown"
    if user_id in user_data:
        current_state = user_data[user_id].get('state', 'No state')
    
    # Get available variables if any
    available_vars = "None"
    if user_id in user_data and 'numeric_vars' in user_data[user_id]:
        available_vars = ", ".join(user_data[user_id]['numeric_vars'])
    
    # Get selected variables if any
    selected_vars = []
    if user_id in user_data:
        if 'dependent_vars' in user_data[user_id]:
            selected_vars.append(f"Dependent: {', '.join(user_data[user_id]['dependent_vars'])}")
        if 'independent_vars' in user_data[user_id]:
            selected_vars.append(f"Independent: {', '.join(user_data[user_id]['independent_vars'])}")
        if 'subgroups' in user_data[user_id]:
            selected_vars.append(f"Subgroups: {', '.join(user_data[user_id]['subgroups'])}")
    
    debug_info = (
        "🔍 *Debug Information*\n\n"
        f"Current State: {current_state}\n\n"
        f"Available Variables:\n{available_vars}\n\n"
        f"Selected Variables:\n" + ("\n".join(selected_vars) if selected_vars else "None") + "\n\n"
        f"User Data Keys: {list(user_data.get(user_id, {}).keys())}\n\n"
        "Use /start to begin a new session if needed."
    )
    
    await update.message.reply_text(debug_info, parse_mode='Markdown')

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handler for /cancel command - cancels the current operation"""
    user_id = update.effective_user.id
    if user_id in user_data and "file_path" in user_data[user_id]:
        try:
            os.unlink(user_data[user_id]["file_path"])
        except:
            pass
        del user_data[user_id]
    
    await update.message.reply_text("Операция отменена. Отправьте /start для начала новой.")
    return ConversationHandler.END

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handler for uploaded documents"""
    user_id = update.effective_user.id
    logger.debug(f"handle_document called for user {user_id}")
    logger.debug(f"Current user_data: {user_data.get(user_id, 'No data')}")
    
    # Check if user has an existing file being processed
    if user_id in user_data and "file_path" in user_data[user_id]:
        try:
            os.unlink(user_data[user_id]["file_path"])
        except:
            pass
        del user_data[user_id]
    
    # Validate file format
    document = update.message.document
    if not document.file_name.lower().endswith('.sav'):
        await update.message.reply_text(
            "❌ Invalid file format!\n\n"
            "Please send a file in SPSS format (.sav extension).\n"
            "If you need help, use the /help command."
        )
        return UPLOAD_FILE
    
    # Check file size (limit to 50MB)
    if document.file_size > 50 * 1024 * 1024:  # 50MB in bytes
        await update.message.reply_text(
            "❌ File too large!\n\n"
            "Maximum file size is 50MB. Please send a smaller file."
        )
        return UPLOAD_FILE
    
    await update.message.reply_text(
        "📥 Receiving your file...\n\n"
        "I'll validate the file format and structure. This might take a moment."
    )
    
    try:
        # Download file
        file = await document.get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.sav') as temp_file:
            await file.download_to_drive(custom_path=temp_file.name)
            user_data[user_id] = {"file_path": temp_file.name}

        await update.message.reply_text(
            "✅ File received successfully!\n"
            "Starting data processing..."
        )

        # Delegate processing to a single robust function
        return await process_file(update, context, user_id)
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}")
        await update.message.reply_text(f"Произошла ошибка при обработке файла: {str(e)}")
        # Удаляем временный файл
        if user_id in user_data and "file_path" in user_data[user_id]:
            try:
                os.unlink(user_data[user_id]["file_path"])
            except:
                pass
            del user_data[user_id]
        return ConversationHandler.END

async def show_variable_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show help message for the current variable type"""
    user_id = update.effective_user.id
    
    # Get current variable type and state
    var_type = context.user_data.get('current_var_type')
    current_state = context.user_data.get('state')
    
    if var_type and var_type in VARIABLE_HELP:
        # Send help message with the same keyboard
        if current_state == SELECT_SUBGROUPS:
            keyboard = get_action_keyboard(['skip', 'cancel'], help_button=True)
        else:
            keyboard = get_action_keyboard(['cancel'], help_button=True)
            
        await update.message.reply_text(
            VARIABLE_HELP[var_type],
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Return to the same state
        return current_state
    
    return current_state

async def handle_dependent_vars(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle selection of dependent variables"""
    user_id = update.effective_user.id
    logger.debug("handle_dependent_vars called")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Message text: {update.message.text}")
    logger.debug(f"Current user_data: {user_data.get(user_id, 'No data')}")
    logger.debug(f"Current state: {context.user_data.get('state', 'No state')}")

    # Get user data safely
    user_data_copy = get_user_data(user_id)
    if not user_data_copy:
        logger.error(f"User {user_id} not found in user_data")
        await update.message.reply_text("Session expired. Please start over with /start")
        return ConversationHandler.END

    if "numeric_vars" not in user_data_copy:
        logger.error(f"numeric_vars not found for user {user_id}")
        await update.message.reply_text("Session data corrupted. Please start over with /start")
        cleanup_user_data(user_id)
        return ConversationHandler.END
    
    # Process variable selection
    try:
        text = update.message.text.strip()
        logger.info(f"Received text: {text}")
        
        # Check for cancel (case insensitive)
        if re.match(r'^cancel$', text, re.IGNORECASE):
            logger.info(f"User {user_id} cancelled dependent variable selection")
            await update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
            await cancel(update, context)
            return ConversationHandler.END
        
        if not text:
            logger.error("Received empty text")
            await update.message.reply_text("Please send variable names separated by commas.")
            return SELECT_DEPENDENT
            
        selected_vars = parse_variable_list(text)
        logger.info(f"Parsed selected vars: {selected_vars}")
        
        if not selected_vars:
            logger.error("No valid variables found after parsing")
            await update.message.reply_text(
                "❌ No valid variables found.\n"
                "Please type variable names separated by commas.\n"
                "Example: `var1, var2, var3`"
            )
            return SELECT_DEPENDENT
        
        numeric_vars = user_data_copy["numeric_vars"]
        logger.info(f"Available numeric vars: {numeric_vars}")
        
        # Validate selections with case-insensitive matching
        matched_vars, invalid_vars = match_variables_case_insensitive(selected_vars, numeric_vars)
        
        if invalid_vars:
            # Don't list all variables if there are too many
            if len(numeric_vars) > 50:
                error_msg = (
                    f"❌ Invalid variable(s): {', '.join(invalid_vars)}\n\n"
                    f"Variable names are case-insensitive.\n"
                    f"There are {len(numeric_vars)} available numeric variables.\n\n"
                    "Please check the variable names and try again.\n"
                    "Example: `var1, var2, var3`"
                )
            else:
                error_msg = (
                    f"❌ Invalid variable(s): {', '.join(invalid_vars)}\n\n"
                    "Please select from the available numeric variables:\n"
                    "```\n"
                    f"{', '.join(numeric_vars)}\n"
                    "```"
                )
            logger.info(f"Invalid variables found: {invalid_vars}")
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            return SELECT_DEPENDENT
    except Exception as e:
        logger.error(f"Error in handle_dependent_vars: {str(e)}")
        await update.message.reply_text(
            "❌ Error processing your selection. Please try again with the format: var1, var2, var3"
        )
        return SELECT_DEPENDENT
    
    # Store selections safely - use matched_vars with correct case
    user_data_copy["dependent_vars"] = matched_vars
    logger.info(f"Stored dependent vars with correct case: {matched_vars}")
    user_data_copy["state"] = SELECT_INDEPENDENT
    update_user_data(user_id, user_data_copy)
    
    # Calculate remaining variables - use matched_vars with correct case
    remaining_vars = [v for v in numeric_vars if v not in matched_vars]
    
    # Ask for independent variables using the formatting function
    message = format_variable_list_message(remaining_vars, 'independent')
    
    # Create keyboard with Cancel and Help buttons
    keyboard = get_action_keyboard(['cancel'], help_button=True)
    
    # Store current variable type for help handler
    context.user_data['current_var_type'] = 'independent'
    
    await update.message.reply_text(
        message,
        reply_markup=keyboard,
        parse_mode='Markdown'
    )
    
    # Update conversation state
    context.user_data['state'] = SELECT_INDEPENDENT
    return SELECT_INDEPENDENT

async def handle_independent_vars(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle selection of independent variables"""
    user_id = update.effective_user.id
    logger.info("=" * 50)
    logger.info(f"Handling independent vars for user {user_id}")
    logger.info(f"Message text: {update.message.text}")
    logger.info(f"Current conversation state: {context.user_data}")
    logger.info("=" * 50)

    # Get user data safely
    user_data_copy = get_user_data(user_id)
    if not user_data_copy:
        logger.error(f"User {user_id} not found in user_data")
        await update.message.reply_text("Session expired. Please start over with /start")
        return ConversationHandler.END

    required_fields = ["numeric_vars", "dependent_vars"]
    missing_fields = [f for f in required_fields if f not in user_data_copy]
    if missing_fields:
        logger.error(f"Missing required fields for user {user_id}: {missing_fields}")
        await update.message.reply_text("Session data corrupted. Please start over with /start")
        cleanup_user_data(user_id)
        return ConversationHandler.END

    try:
        text = update.message.text.strip()
        logger.info(f"Received text: {text}")
        
        # Check for cancel (case insensitive)
        if re.match(r'^cancel$', text, re.IGNORECASE):
            logger.info(f"User {user_id} cancelled independent variable selection")
            await update.message.reply_text("Operation cancelled.", reply_markup=ReplyKeyboardRemove())
            await cancel(update, context)
            return ConversationHandler.END
        
        if not text:
            logger.error("Received empty text")
            await update.message.reply_text("Please send variable names separated by commas.")
            return SELECT_INDEPENDENT
            
        selected_vars = parse_variable_list(text)
        logger.info(f"Parsed selected vars: {selected_vars}")
        
        if not selected_vars:
            logger.error("No valid variables found after parsing")
            await update.message.reply_text(
                "❌ No valid variables found.\n"
                "Please type variable names separated by commas.\n"
                "Example: `var1, var2, var3`"
            )
            return SELECT_INDEPENDENT
        
        numeric_vars = user_data_copy["numeric_vars"]
        dependent_vars = user_data_copy["dependent_vars"]
        available_vars = [v for v in numeric_vars if v not in dependent_vars]
        
        # Validate selections with case-insensitive matching
        matched_vars, invalid_vars = match_variables_case_insensitive(selected_vars, available_vars)
        
        if invalid_vars:
            # Don't list all variables if there are too many
            if len(available_vars) > 50:
                error_msg = (
                    f"❌ Invalid or already selected variable(s): {', '.join(invalid_vars)}\n\n"
                    f"Variable names are case-insensitive.\n"
                    f"There are {len(available_vars)} available variables.\n\n"
                    "Please check the variable names and try again.\n"
                    "Example: `var1, var2, var3`"
                )
            else:
                error_msg = (
                    f"❌ Invalid or already selected variable(s): {', '.join(invalid_vars)}\n\n"
                    "Please select from the available variables:\n"
                    "```\n"
                    f"{', '.join(available_vars)}\n"
                    "```"
                )
            logger.info(f"Invalid variables found: {invalid_vars}")
            await update.message.reply_text(error_msg, parse_mode='Markdown')
            return SELECT_INDEPENDENT
        
        if len(matched_vars) < 2:
            if len(available_vars) > 50:
                await update.message.reply_text(
                    f"❌ Please select at least 2 independent variables.\n\n"
                    f"You have {len(available_vars)} variables available."
                )
            else:
                await update.message.reply_text(
                    "❌ Please select at least 2 independent variables.\n\n"
                    "Available variables:\n" +
                    ", ".join(available_vars)
                )
            return SELECT_INDEPENDENT
        
        # Store selections safely - use matched_vars with correct case
        user_data_copy["independent_vars"] = matched_vars
        logger.info(f"Stored independent vars with correct case: {matched_vars}")
        user_data_copy["state"] = SELECT_SUBGROUPS
        update_user_data(user_id, user_data_copy)
        
        # Calculate remaining variables - use matched_vars with correct case
        remaining_vars = [v for v in numeric_vars 
                        if v not in dependent_vars and v not in matched_vars]
        
        # Ask for subgroup variables using the formatting function
        message = format_variable_list_message(remaining_vars, 'subgroups')
        
        # Create keyboard with Skip, Cancel and Help buttons
        keyboard = get_action_keyboard(['skip', 'cancel'], help_button=True)
        
        # Store current variable type for help handler
        context.user_data['current_var_type'] = 'subgroups'
        
        await update.message.reply_text(
            message,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Update conversation state
        context.user_data['state'] = SELECT_SUBGROUPS
        return SELECT_SUBGROUPS
        
    except Exception as e:
        logger.error(f"Error in handle_independent_vars: {str(e)}")
        # Create keyboard with Cancel and Help buttons
        keyboard = get_action_keyboard(['cancel'], help_button=True)
        await update.message.reply_text(
            "❌ Error processing your selection.\n"
            "Please type variable names separated by commas:\n"
            "```\n"
            "var1, var2, var3\n"
            "```\n"
            "Press ❓ Help for more information",
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        return SELECT_INDEPENDENT

async def handle_subgroups(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle selection of subgroup variables"""
    user_id = update.effective_user.id
    logger.info("=" * 50)
    logger.info(f"Handling subgroups for user {user_id}")
    logger.info(f"Message text: {update.message.text}")
    logger.info(f"Current conversation state: {context.user_data}")
    logger.info("=" * 50)

    # Get user data safely
    user_data_copy = get_user_data(user_id)
    if not user_data_copy:
        logger.error(f"User {user_id} not found in user_data")
        await update.message.reply_text("Session expired. Please start over with /start")
        return ConversationHandler.END

    required_fields = ["numeric_vars", "dependent_vars", "independent_vars"]
    missing_fields = [f for f in required_fields if f not in user_data_copy]
    if missing_fields:
        logger.error(f"Missing required fields for user {user_id}: {missing_fields}")
        await update.message.reply_text("Session data corrupted. Please start over with /start")
        cleanup_user_data(user_id)
        return ConversationHandler.END

    try:
        raw_text = update.message.text.strip()
        logger.info(f"Received text: {raw_text}")
        
        # Check for skip or cancel (case insensitive)
        if re.match(r'^(skip|cancel)$', raw_text, re.IGNORECASE):
            # Remove keyboard
            reply_markup = ReplyKeyboardRemove()
            
            if re.match(r'^skip$', raw_text, re.IGNORECASE):
                # User wants to skip subgroups - set empty list and proceed to summary
                user_data_copy["subgroups"] = []
                user_data_copy["state"] = CONFIRM_ANALYSIS
                update_user_data(user_id, user_data_copy)
                logger.info(f"User {user_id} skipped subgroup selection")
            else:  # cancel
                await update.message.reply_text("Operation cancelled.", reply_markup=reply_markup)
                await cancel(update, context)
                return ConversationHandler.END
        else:
            if not raw_text:
                logger.error("Received empty text")
                # Show error and provide buttons again
                keyboard = get_action_keyboard(['skip', 'cancel'], help_button=True)
                await update.message.reply_text(
                    "❌ Please either:\n"
                    "• Type variable names separated by commas\n"
                    "• Press Skip to proceed without subgroups\n"
                    "• Press Cancel to start over",
                    reply_markup=keyboard
                )
                return SELECT_SUBGROUPS
                
            selected_vars = parse_variable_list(raw_text)
            logger.info(f"Parsed selected vars: {selected_vars}")
            
            # Even if parse_variable_list returns empty list due to trailing commas,
            # we should still process the input if it's not completely empty
            if not raw_text.replace(',', '').strip():
                logger.error("No valid variables found after parsing")
                await update.message.reply_text(
                    "❌ No valid variables found.\n"
                    "Please type variable names separated by commas or press Skip.\n"
                    "Example: `var1, var2, var3`"
                )
                return SELECT_SUBGROUPS
            
            numeric_vars = user_data_copy["numeric_vars"]
            used_vars = user_data_copy["dependent_vars"] + user_data_copy["independent_vars"]
            available_vars = [v for v in numeric_vars if v not in used_vars]
            
            # For subgroups, we allow both numeric and categorical variables
            all_available_vars = user_data_copy.get("all_vars", [])
            subgroup_available_vars = [v for v in all_available_vars if v not in used_vars]
            
            # Validate selections with case-insensitive matching
            matched_vars, invalid_vars = match_variables_case_insensitive(selected_vars, subgroup_available_vars)
            
            if invalid_vars:
                # Don't list all variables if there are too many
                available_for_subgroups = [v for v in all_available_vars if v not in used_vars]
                if len(available_for_subgroups) > 50:
                    error_msg = (
                        f"❌ Invalid or already selected variable(s): {', '.join(invalid_vars)}\n\n"
                        f"Variable names are case-insensitive.\n"
                        f"There are {len(available_for_subgroups)} available variables.\n\n"
                        "Please check the variable names and try again, or press Skip.\n"
                        "Example: `var1, var2, var3`"
                    )
                else:
                    error_msg = (
                        f"❌ Invalid or already selected variable(s): {', '.join(invalid_vars)}\n\n"
                        "Please select from the available variables or press Skip:\n"
                        "```\n"
                        f"{', '.join(available_for_subgroups)}\n"
                        "```"
                    )
                logger.info(f"Invalid variables found: {invalid_vars}")
                await update.message.reply_text(error_msg, parse_mode='Markdown')
                return SELECT_SUBGROUPS
                
            # Store selections safely - use matched_vars with correct case
            user_data_copy["subgroups"] = matched_vars
            logger.info(f"Stored subgroup vars with correct case: {matched_vars}")
            user_data_copy["state"] = CONFIRM_ANALYSIS
            update_user_data(user_id, user_data_copy)
        
        # Show analysis summary and ask for confirmation
        summary = (
            "📋 *Analysis Summary*\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "*Selected Variables:*\n\n"
            "🎯 **Dependent Variables**:\n"
            "```\n"
            f"{', '.join(user_data_copy['dependent_vars'])}\n"
            "```\n\n"
            "📊 **Independent Variables**:\n"
            "```\n"
            f"{', '.join(user_data_copy['independent_vars'])}\n"
            "```\n\n"
            "🔍 **Subgroup Variables**:\n"
            "```\n"
            f"{', '.join(user_data_copy['subgroups']) if user_data_copy['subgroups'] else 'None'}\n"
            "```\n\n"
            "*What's Next:*\n"
            "• Press Confirm to start the analysis\n"
            "• Press Cancel to start over\n"
        )
        
        # Create keyboard with Confirm and Cancel buttons
        keyboard = get_action_keyboard(['confirm', 'cancel'])
        
        await update.message.reply_text(
            summary,
            reply_markup=keyboard,
            parse_mode='Markdown'
        )
        
        # Update conversation state
        context.user_data['state'] = CONFIRM_ANALYSIS
        return CONFIRM_ANALYSIS
        
    except Exception as e:
        logger.error(f"Error in handle_subgroups: {str(e)}")
        await update.message.reply_text(
            "❌ Error processing your selection. Please try again with the format: var1, var2, var3 or press 'Skip'"
        )
        return SELECT_SUBGROUPS


async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle analysis confirmation"""
    user_id = update.effective_user.id
    if user_id not in user_data:
        await update.message.reply_text("Session expired. Please start over with /start")
        return ConversationHandler.END
    
    text = update.message.text.strip().lower()
    if text in ['confirm', 'cancel']:
        # Remove keyboard
        reply_markup = ReplyKeyboardRemove()
        
        if text == 'confirm':
            # Start the analysis
            await update.message.reply_text(
                "🔄 Starting Johnson's Relative Weights analysis...\n"
                "This may take a few moments.",
                reply_markup=reply_markup
            )
            await start_analysis(update, context, user_id)
            return ConversationHandler.END
        else:  # cancel
            await update.message.reply_text("Operation cancelled.", reply_markup=reply_markup)
            await cancel(update, context)
            return ConversationHandler.END
    else:
        # Show error and provide buttons again
        keyboard = get_action_keyboard(['confirm', 'cancel'])
        await update.message.reply_text(
            "❌ Invalid response. Please use the buttons below:\n"
            "• Press Confirm to start the analysis\n"
            "• Press Cancel to start over",
            reply_markup=keyboard
        )
        return CONFIRM_ANALYSIS

async def start_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    """Start the Johnson's Relative Weights analysis"""
    try:
        # Lazy import тяжелых библиотек только при обработке
        from weights_handler import WeightsCalculationHandler
        
        file_path = user_data[user_id]["file_path"]
        dependent_vars = user_data[user_id]["dependent_vars"]
        independent_vars = user_data[user_id]["independent_vars"]
        subgroups = user_data[user_id]["subgroups"]
        
        # Initialize weights handler
        handler = WeightsCalculationHandler()
        
        # Validate analysis parameters
        is_valid, error_msg = handler.validate_analysis_parameters(
            input_file=file_path,
            dependent_vars=dependent_vars,
            independent_vars=independent_vars,
            subgroups=subgroups if subgroups else None
        )
        
        if not is_valid:
            await update.message.reply_text(
                f"❌ Validation error: {error_msg}\n"
                "Please check your variable selections and try again."
            )
            return
        
        # Determine analysis type
        analysis_type = "total" if not subgroups else "group"
            
        # Call the analysis function
        result = handler.calculate_weights(
            input_file=file_path,
            dependent_vars=dependent_vars,
            independent_vars=independent_vars,
            analysis_type=analysis_type,
            subgroups=subgroups if subgroups else None
        )
        
        if result['status'] == 'success':
            # Send both Excel and CSV files
            excel_file = result['results']
            csv_file = excel_file.replace('.xlsx', '.csv')
            
            # Send Excel file
            await update.message.reply_document(
                document=open(excel_file, 'rb'),
                caption="✅ Analysis completed successfully! Sending both Excel and CSV files..."
            )
            
            # Send CSV file
            await update.message.reply_document(
                document=open(csv_file, 'rb'),
                caption="Here's your CSV file with the same results."
            )
            
            # Send completion message with next steps
            completion_message = (
                "🎉 Analysis completed successfully!\n\n"
                "Results include calculations using three different imputation methods:\n"
                "- MICE (Multiple Imputation by Chained Equations)\n"
                "- Hybrid approach with baseline\n"
                "- Simple mean imputation\n\n"
                "For detailed description of both methods, see Multiple Imputations Readme.txt\n\n"
                "Next steps:\n"
                "1. Check both Excel and CSV files for your results\n"
                "2. Use /start to begin a new analysis\n"
                "3. Or use /cancel to end the current session"
            )
            await update.message.reply_text(completion_message)
        else:
            await update.message.reply_text(
                f"❌ Error during analysis: {result['message']}\n"
                "Please check your variable selections and try again."
            )
    except Exception as e:
        await update.message.reply_text(
            f"❌ Error during analysis: {str(e)}\n"
            "Please try again with different variables."
        )
    finally:
        # Clean up
        if user_id in user_data:
            if "file_path" in user_data[user_id]:
                try:
                    os.unlink(user_data[user_id]["file_path"])
                except:
                    pass
            del user_data[user_id]

async def process_file(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> None:
    """Process the uploaded file and calculate Johnson's weights"""
    from spss_handlers import validate_spss_file, read_spss_with_fallbacks, SPSSReadError
    from file_handlers.repair_handler import SPSSFileRepairHandler
    
    file_path = user_data[user_id]["file_path"]
    
    try:
        # First validate the file
        is_valid, message, file_info = validate_spss_file(file_path)
        
        needs_repair = file_info and file_info.get('needs_repair', False)
        if needs_repair or not is_valid:
            # Inform user about repair attempt
            await update.message.reply_text(
                "⚠️ Having trouble reading the file. Let me try to fix it...\n"
                "This might take a moment."
            )
            
            # Attempt repair
            repair_handler = SPSSFileRepairHandler(file_path)
            df, meta, repair_attempts = repair_handler.attempt_repair()
            
            if df is None:
                # If repair failed, show what was tried
                error_msg = (
                    "❌ Unable to read the file, even after trying these fixes:\n\n"
                )
                for attempt in repair_attempts:
                    status = "✅" if attempt['success'] else "❌"
                    error_msg += f"{status} {attempt['strategy']}: {attempt['details']}\n"
                    if not attempt['success'] and 'error' in attempt:
                        error_msg += f"   Error: {attempt['error']}\n"
                
                error_msg += (
                    "\nPlease check if:\n"
                    "1. The file is a valid SPSS database\n"
                    "2. The file isn't corrupted\n"
                    "3. Try exporting it again from your SPSS software\n\n"
                    "You can also try sending a different file."
                )
                
                await update.message.reply_text(error_msg)
                return ConversationHandler.END
            else:
                # If repair succeeded, inform user and continue
                await update.message.reply_text(
                    "✅ Successfully fixed and loaded the file!\n"
                    "Continuing with the analysis..."
                )
        else:
            # Try reading the file normally if no repair needed
            try:
                df, meta = read_spss_with_fallbacks(file_path)
            except SPSSReadError as e:
                # If normal reading fails, try repair as fallback
                await update.message.reply_text(
                    "⚠️ Having trouble reading the file. Let me try to fix it...\n"
                    "This might take a moment."
                )
                
                repair_handler = SPSSFileRepairHandler(file_path)
                df, meta, repair_attempts = repair_handler.attempt_repair()
                
                if df is None:
                    error_msg = (
                        "❌ Unable to read the file, even after trying these fixes:\n\n"
                    )
                    for attempt in repair_attempts:
                        status = "✅" if attempt['success'] else "❌"
                        error_msg += f"{status} {attempt['strategy']}: {attempt['details']}\n"
                        if not attempt['success'] and 'error' in attempt:
                            error_msg += f"   Error: {attempt['error']}\n"
                    
                    error_msg += (
                        "\nPlease check if:\n"
                        "1. The file is a valid SPSS database\n"
                        "2. The file isn't corrupted\n"
                        "3. Try exporting it again from your SPSS software\n\n"
                        "You can also try sending a different file."
                    )
                    
                    await update.message.reply_text(error_msg)
                    return ConversationHandler.END
                else:
                    await update.message.reply_text(
                        "✅ Successfully fixed and loaded the file!\n"
                        "Continuing with the analysis..."
                    )
            
        # Initialize validation results
        validation_results = []
        
        # 1. Sample size validation
        min_sample_size = 100
        validation_results.append(
            f"{'✅' if df.shape[0] >= min_sample_size else '❌'} "
            f"Sample size: {df.shape[0]} rows (minimum required: {min_sample_size})"
        )
        
        # 2. Variable type validation
        numeric_vars = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        non_numeric_vars = [col for col in df.columns if col not in numeric_vars]
        
        validation_results.append(
            f"{'✅' if len(numeric_vars) >= 3 else '❌'} "
            f"Found {len(numeric_vars)} numeric variables (minimum required: 3)"
        )
        
        # 3. File format info
        if file_info:
            format_type = file_info.get('type', 'standard')
            validation_results.append(
                f"\n📄 File Information:\n"
                f"- Format: {format_type.title()}\n"
                f"- Encoding: {file_info.get('encoding', 'unknown')}"
            )
        
        # 4. Special codes analysis
        special_codes_vars = []
        for var in numeric_vars:
            unique_vals = set(df[var].dropna().unique())
            if any(code in unique_vals for code in [98, 99]):
                special_codes_vars.append(var)
        
        if special_codes_vars:
            validation_results.append("\nVariables with special codes (98, 99):")
            for var in special_codes_vars[:5]:
                count = df[var].isin([98, 99]).sum()
                validation_results.append(f"- {var}: {count} special values")
            if len(special_codes_vars) > 5:
                validation_results.append("...")
        
        # 5. Variable variance check
        zero_variance_vars = []
        for var in numeric_vars:
            if df[var].nunique() <= 1:
                zero_variance_vars.append(var)
        
        if zero_variance_vars:
            validation_results.append("\n⚠️ Variables with no variance (will be excluded):")
            for var in zero_variance_vars[:5]:
                validation_results.append(f"- {var}")
            if len(zero_variance_vars) > 5:
                validation_results.append("...")
        
        # Remove zero variance variables
        numeric_vars = [var for var in numeric_vars if var not in zero_variance_vars]
        
        # Store metadata in user_data for future use
        user_data[user_id] = {
            "numeric_vars": numeric_vars,
            "categorical_vars": non_numeric_vars,  # Store categorical variables
            "meta": meta,
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "file_info": file_info,
            "file_path": file_path,  # Keep the file path
            "all_vars": df.columns.tolist()
        }
        
        # Log for debugging
        logger.info(f"Stored user data for user {user_id}: {user_data[user_id]}")
        logger.info(f"Available numeric variables: {numeric_vars}")
        
        # Prepare report
        report = (
            "📊 Data Structure Validation Report\n\n" + 
            "\n".join(validation_results)
        )
        
        # Check if file is usable
        if len(numeric_vars) >= 3 and df.shape[0] >= min_sample_size:
            # First, send the validation report (with length check)
            if len(report) > 3000:
                # Report is too long, send a shortened version
                short_report = (
                    "📊 Data Structure Validation Report\n\n"
                    f"✅ Sample size: {df.shape[0]} rows\n"
                    f"✅ Found {len(numeric_vars)} numeric variables\n"
                    f"✅ File format: {file_info.get('type', 'standard').title()}\n"
                    f"✅ Encoding: {file_info.get('encoding', 'unknown')}\n\n"
                    "File validated successfully!"
                )
                await update.message.reply_text(short_report)
            else:
                await update.message.reply_text(report)
            
            # Then, send the variable selection message using formatting function
            var_selection_msg = format_variable_list_message(numeric_vars, 'dependent')
            
            # Create keyboard with Cancel and Help buttons
            keyboard = get_action_keyboard(['cancel'], help_button=True)
            
            # Store current variable type for help handler
            context.user_data['current_var_type'] = 'dependent'
            
            await update.message.reply_text(
                var_selection_msg,
                reply_markup=keyboard,
                parse_mode='Markdown'
            )
            
            # Store the state in both context and user_data
            context.user_data['state'] = SELECT_DEPENDENT
            user_data[user_id]['state'] = SELECT_DEPENDENT
            
            logger.debug("Transitioning to SELECT_DEPENDENT state")
            logger.debug(f"User ID: {user_id}")
            logger.debug(f"Current user_data: {user_data[user_id]}")
            logger.debug(f"Current state: {context.user_data}")
            logger.debug(f"Available states: {[UPLOAD_FILE, SELECT_DEPENDENT, SELECT_INDEPENDENT, SELECT_SUBGROUPS, CONFIRM_ANALYSIS]}")
            logger.debug(f"SELECT_DEPENDENT value: {SELECT_DEPENDENT}")
            logger.debug(f"Returning state: {SELECT_DEPENDENT}")
            
            return SELECT_DEPENDENT
        else:
            report += (
                "\n\n❌ Cannot Proceed:\n"
                "The file does not meet the minimum requirements:\n"
                "- At least 3 numeric variables\n"
                "- At least 100 rows of data\n\n"
                "Please check your file and try again."
            )
            await update.message.reply_text(report)
            return ConversationHandler.END
    except Exception as e:
        error = e  # Store error for finally block
        error_message = (
            "❌ Error Processing File\n\n"
            f"Error: {str(e)}\n\n"
            "Please check your file format and try again.\n"
            "Use /help for file requirements."
        )
        logger.error(f"Error in process_file: {str(e)}")
        await update.message.reply_text(error_message)
        return ConversationHandler.END
    finally:
        # Clean up temporary file if there was an error
        if "error" in locals():
            try:
                os.unlink(file_path)
            except:
                pass
            if user_id in user_data:
                del user_data[user_id]

def main() -> None:
    """Start the bot"""
    # Skip lock file in webhook mode (Serverless Containers)
    webhook_url = os.getenv('WEBHOOK_URL')
    if not webhook_url:
        acquire_lock()  # Only use lock in polling mode
    
    logger.info(f"Bot starting with PID={os.getpid()}")
    application = Application.builder().token(TOKEN).build()
    
    # Log available states
    logger.info("=" * 50)
    logger.info("Initializing conversation handler")
    logger.info(f"Available states: {[UPLOAD_FILE, SELECT_DEPENDENT, SELECT_INDEPENDENT, SELECT_SUBGROUPS, CONFIRM_ANALYSIS]}")
    logger.info(f"State values: UPLOAD_FILE={UPLOAD_FILE}, SELECT_DEPENDENT={SELECT_DEPENDENT}")
    logger.info("=" * 50)

    # Настройка обработчика диалога
    async def prompt_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Please upload an SPSS .sav file to begin, or use /help.")
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start)
        ],
        states={
            UPLOAD_FILE: [
                MessageHandler(filters.Document.ALL, handle_document),
                MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_upload),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel)
            ],
            SELECT_DEPENDENT: [
                MessageHandler(filters.Regex(r'^❓ Help$'), show_variable_help),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_dependent_vars),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
                CommandHandler("start", start)
            ],
            SELECT_INDEPENDENT: [
                MessageHandler(filters.Regex(r'^❓ Help$'), show_variable_help),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_independent_vars),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
                CommandHandler("start", start)
            ],
            SELECT_SUBGROUPS: [
                MessageHandler(filters.Regex(r'^❓ Help$'), show_variable_help),
                MessageHandler(filters.TEXT & filters.Regex(r'^[Ss][Kk][Ii][Pp]$'), handle_subgroups),
                MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(r'^[Ss][Kk][Ii][Pp]$'), handle_subgroups),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
                CommandHandler("start", start)
            ],
            CONFIRM_ANALYSIS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_confirmation),
                CommandHandler("help", help_command),
                CommandHandler("cancel", cancel),
                CommandHandler("start", start)
            ],
        },
        fallbacks=[
            CommandHandler("cancel", cancel),
            CommandHandler("start", start),
            CommandHandler("help", help_command)
        ],
        allow_reentry=True,
        name="main_conversation",
        persistent=False,
        per_message=False,
        per_chat=True,
        map_to_parent=None
    )
    
    # Add a message handler for logging all messages
    async def log_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("=" * 50)
        logger.info("UNHANDLED Message received:")
        logger.info(f"From user: {update.effective_user.id}")
        logger.info(f"Text: {update.message.text}")
        logger.info(f"Current user_data: {user_data.get(update.effective_user.id, 'No user data')}")
        logger.info(f"Current conversation state: {context.user_data}")
        logger.info(f"Message type: {update.message.content_type}")
        logger.info(f"All handlers: {[h.__class__.__name__ for h in application.handlers[0]]}")
        logger.info(f"Active conversation: {conv_handler.check_update(update)}")
        logger.info("=" * 50)
        
        # Send feedback to user that message was unhandled
        await update.message.reply_text(
            "⚠️ Your message wasn't handled by any active command.\n"
            "You might be in the wrong state or the session might have expired.\n"
            "Use /start to begin a new session or /help for guidance."
        )
    
    # Add debug handler to log all updates before they reach handlers
    async def debug_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        logger.info("=" * 50)
        logger.info("DEBUG: Update received")
        try:
            if update.message and update.message.document:
                utype = 'document'
            elif update.message and update.message.text:
                utype = 'text'
            else:
                utype = 'other'
        except Exception:
            utype = 'unknown'
        logger.info(f"Update type: {utype}")
        logger.info(f"From user: {update.effective_user.id}")
        if update.message and update.message.text:
            logger.info(f"Text: {update.message.text}")
        logger.info(f"Current conversation state: {context.user_data.get('state', 'No state')}")
        # Removed conv_handler.get_conversations() as it's not available in this PTB version
        logger.info("=" * 50)
        return
    
    # Add handlers in order of priority
    application.add_handler(MessageHandler(filters.ALL, debug_handler, block=False), -1)  # Debug handler first, non-blocking
    application.add_handler(conv_handler)  # Conversation handler second
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, log_message))  # Logging handler for unhandled messages
    
    # webhook_url уже получен в начале main()
    if webhook_url:
        # Режим webhook для production (Yandex Cloud)
        logger.info(f"Starting bot in webhook mode: {webhook_url}")
        
        # Настройка webhook
        application.run_webhook(
            listen="0.0.0.0",
            port=8080,
            url_path=TOKEN,
            webhook_url=f"{webhook_url}/{TOKEN}",
            drop_pending_updates=True
        )
    else:
        # Режим polling для локальной разработки
        logger.info("Starting bot in polling mode (local development)")
        application.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
