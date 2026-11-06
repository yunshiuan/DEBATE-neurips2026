"""Original training-context formatting; orchestration is in pipelines.prepare_sft."""
import pandas as pd
import json
import os
import ast
import re
import sys
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parents[1])
DATA_DIR = "."
INTERMEDIATE_DIR = "examples"
MISSING_REJUDGE_RECORDS = []
_MISSING_REJUDGE_KEYS = set()
POST_VALIDITY_CHECK = None

def is_post_opinion_valid_rejudge(dp, player):
    if POST_VALIDITY_CHECK is None:
        raise RuntimeError("An explicit post-opinion validity policy is required")
    return POST_VALIDITY_CHECK(dp, player)

def record_missing_rejudge(rejudge_path: str, dp: str, player: str):
    key = (rejudge_path, str(dp), str(player))
    if key in _MISSING_REJUDGE_KEYS:
        return
    _MISSING_REJUDGE_KEYS.add(key)
    MISSING_REJUDGE_RECORDS.append({
        "rejudge_path": rejudge_path,
        "data_prefix": str(dp),
        "worker_id": str(player),
    })


def get_demographic_background(user_data, player_column_name):
    """
    Retrieves the demographic background of each player from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        api_type (str): The type of API to use for the agents (openai or huggingface).
        model_name (str): The name of the model to use for generating responses.

    Returns:
        dict: A dictionary where keys are player names and values are their demographic backgrounds.
    """
    # Read the markdown file once and store its content
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "demographics.md"), "r") as f:
        demographic_template = f.read()
    
    player_list = user_data[player_column_name].dropna().unique()
    demographic_backgrounds = {}
    df_fields = ["age", "gender", "education", "ethnicity", "income", "politicalIdentity", "politicalViews", "childrenSchool", "residence", "maritalStatus", "bibleBelief", "evangelical", "religion", "occupation"]
    for player in player_list:
        player_survey = user_data[(user_data['event_type'] == 'exit_survey') & (user_data[player_column_name] == player)]
        player_survey_fields = player_survey['field'].tolist()
        if len(player_survey) == 0:
            demographic_backgrounds[player] = ""
            print(f"[WARN] Player {player} missing demographic data, skip demographics in simulation", file=sys.stderr)
            continue
        elif len(player_survey) < len(df_fields):
            print(f"[WARN] Player {player} missing some demographic data, plug in 'unknown' in demographics of simulation", file=sys.stderr)
        player_data = {field: player_survey[player_survey['field'] == field]['text'].iloc[0] if field in player_survey_fields else "unknown" for field in df_fields}
        try:
            player_data["ethnicity"] = ", ".join(ast.literal_eval(player_data["ethnicity"]))
        except (ValueError, SyntaxError, TypeError):
            print(f"[WARN] Player {player} has invalid ethnicity {player_data['ethnicity']}, skip formatting it", file=sys.stderr)
        camel_to_snake = lambda s: re.sub(r'([A-Z])', r'_\1', s).upper()  # for example, 'politicalIdentity' -> 'POLITICAL_IDENTITY'
        template_fill = {camel_to_snake(field): player_data[field] for field in df_fields}
        demographic_background = demographic_template.format(**template_fill)  # format the demographic template with the player's data
        demographic_backgrounds[player] = demographic_background

    return demographic_backgrounds


def get_system_prompt(persona: str, topic: str, initial_opinion_likert: str, initial_opinion_verbal: str, player_name: str):
    """
    Generates the system prompt for the chatbot based on the persona, topic, initial opinion, and player name.

    Args:
        persona (str): The persona of the player.
        topic (str): The topic of the conversation.
        initial_opinion_likert (str): The initial opinion of the player on a Likert-scale.
        initial_opinion_verbal (str): The verbal explanation of the initial opinion of the player.
        player_name (str): The name of the player.
    Returns:
        str: The system prompt for the chatbot.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "persona.md"), "r") as f:
        sys_prompt = f.read()
    sys_prompt = sys_prompt.format(
        AGENT_PERSONA=persona,
        AGENT_NAME=player_name,
        TOPIC=topic,
        INITIAL_OPINION_LIKERT=initial_opinion_likert,
        INITIAL_OPINION_VERBAL=initial_opinion_verbal
    )
    sys_prompt = sys_prompt.replace('\n', ' ')
    return sys_prompt


def get_tweet(user_data, sender, recipient):
    """
    Retrieves a tweet from the user_data DataFrame based on the sender and recipient.

    Args:
        user_data (pd.DataFrame): The DataFrame containing tweet data.
        sender (str): The name or identifier of the sender.
        recipient (str): The name or identifier of the recipient.

    Returns:
        tuple: A tuple containing (tweet_text, round_number).
             Returns the longest matching tweet if multiple matches are found.
    """
    user_data = user_data.copy()
    user_data['sender_id'] = user_data['sender_id'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else x)
    user_data['recipient_id'] = user_data['recipient_id'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else x)
    
    matching_tweets = user_data[(user_data["sender_id"] == str(sender)) & (user_data["recipient_id"] == str(recipient)) & 
                               (user_data["event_type"] == "tweet")]
    
    if len(matching_tweets) == 0:
        print(f"[WARN] No tweet found for sender {sender} and recipient {recipient}")
        return "", 0
    elif len(matching_tweets) == 1:
        tweet = matching_tweets["text"].iloc[0]
        round_number = matching_tweets["chat_round_order"].iloc[0]
    else:
        # If multiple matches, keep the longest one
        longest_text_idx = matching_tweets['text'].str.len().idxmax()
        tweet = matching_tweets.loc[longest_text_idx, 'text']
        round_number = matching_tweets.loc[longest_text_idx, 'chat_round_order']
        print(f"[INFO] Multiple tweets found for sender {sender} and recipient {recipient}, keeping the longest one")
    
    return tweet, round_number


def get_pairs(user_data, round_number):
    """
    Retrieves the pairs of players for a given round number.

    Args:
        user_data (pd.DataFrame): The DataFrame containing message data.
        round_number (int): The round number to retrieve pairs for.
        agents_v2 (list): The list of agents.
    Returns:
        list: A list of tuples, where each tuple contains (sender_agent, recipient_agent).
              The list represents the pairs of agents for the specified round.
    """
    
    pairs = user_data[(user_data["chat_round_order"] == round_number) & (user_data["event_type"] != "tweet")][["sender_id", "recipient_id"]].dropna().drop_duplicates().values.tolist()
    pairs = list(set(tuple(sorted(pair)) for pair in pairs))
    return pairs


def get_conversation_history(user_data, player1, players_list, all_messages=False, validOnly=True):
    """
    Retrieves the conversation history between two players from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing message data.
        player1 (str): The name or identifier of the first player.
        players_list (list): The list of players to retrieve conversation history for.
        all_messages (bool): If True, retrieve all messages. If False, retrieve only num_messages messages.

    Returns:
        list: A list of tuples, where each tuple contains (sender, message_text).
              The list represents the conversation history between the two players.
    """
    # The number of messages are sent and received (tweet and initial opinion not included)
    conversation_history = {player1: {}}
    user_data = user_data.copy()
    # Drop rows with NA values in sender_id and recipient_id columns
    user_data = user_data.dropna(subset=['sender_id', 'recipient_id'])
    # Cast sender_id and recipient_id from float to int then to str
    user_data['sender_id'] = user_data['sender_id'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else x)
    user_data['recipient_id'] = user_data['recipient_id'].apply(lambda x: str(int(x)) if isinstance(x, float) and not pd.isna(x) else x)
    for player2 in players_list:
        conversation_history[player1][player2] = []
        sent_messages = user_data[
            (user_data["event_type"] == "message_sent") & 
            (((user_data["sender_id"] == str(player1)) & (user_data["recipient_id"] == str(player2))) | 
            ((user_data["sender_id"] == str(player2)) & (user_data["recipient_id"] == str(player1))))
        ]
        # if validOnly:
        #     sent_messages = sent_messages[sent_messages['validity'] == 'VALID']

        for index, row in sent_messages.iterrows():
            current_player = player1 if row["sender_id"] == player1 or row['worker_id'] == player1 else player2
            current_message = row["text"]
            validity = row["validity"]
            round_number = row["chat_round_order"]
            if pd.isna(current_message) or current_message.strip() == "":
                continue
            
            conversation_history[player1][player2].append((current_player, current_message, validity, round_number))
        
    # {player1: {player2: [(current_player, current_message), ...], player3: [(current_player, current_message), ...]}}
    return conversation_history


def get_initial_opinion(user_data, player, player_column_name):
    """
    Retrieves the initial opinion of a player from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        player (str): The name or identifier of the player.

    Returns:
        str: The initial opinion of the player.
    """
    opinion_slider_value_mapping = {
    "1": "Certainly disagree",
    "2": "Probably disagree",
    "3": "Lean disagree",
    "4": "Lean agree",
    "5": "Probably agree",
    "6": "Certainly agree",
    "unknown": "unknown"
}
    matching_opinions = user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Initial Opinion")]
    
    if len(matching_opinions) == 0:
        print(f"[WARN] No initial opinion found for player {player}")
        return "unknown", opinion_slider_value_mapping["unknown"]
    elif len(matching_opinions) == 1:
        initial_opinion = matching_opinions["text"].iloc[0]
        try:
            slider_value = str(int(matching_opinions["sliderValue"].iloc[0]))
        except:
            slider_value = "unknown"
    else:
        # If multiple matches, keep the longest one
        longest_text_idx = matching_opinions['text'].str.len().idxmax()
        initial_opinion = matching_opinions.loc[longest_text_idx, 'text']
        try:
            slider_value = str(int(matching_opinions.loc[longest_text_idx, 'sliderValue']))
        except:
            slider_value = "unknown"
        print(f"[INFO] Multiple initial opinions found for player {player}, keeping the longest one")
    
    return initial_opinion, opinion_slider_value_mapping[slider_value]


def get_generate_message_prompt(first_agent_name, second_agent_name, topic):
    """
    Generates the prompt for generating a message.

    Args:
        first_agent_name (str): The name or identifier of the first agent.
        second_agent_name (str): The name or identifier of the second agent.
        topic (str): The topic of the conversation.
        max_length (int): The maximum length of the message.
    Returns:
        str: The prompt for generating a message.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "generate_message.md")) as f:
        prompt_instructions = f.read()
        prompt = prompt_instructions.format(
            # FIRST_AGENT_NAME=first_agent_name,
            SECOND_AGENT_NAME=second_agent_name,
            TOPIC=topic
        )
    prompt = prompt.replace('\n', ' ')
    return prompt


def get_generate_tweet_prompt(first_agent_name, second_agent_name, topic):
    """
    Generates the prompt for generating a tweet.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "generate_tweet.md")) as f:
        prompt_instructions = f.read()
        prompt = prompt_instructions.format(
            SECOND_AGENT_NAME=second_agent_name,
            TOPIC=topic
        )
    prompt = prompt.replace('\n', ' ')
    return prompt


def get_change_round_prompt(previous_player, next_player):
    """
    Generates the prompt for changing the round.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "change_round.md")) as f:
        prompt_instructions = f.read()
        prompt = prompt_instructions.format(
            PREVIOUS_AGENT_NAME=previous_player,
            NEXT_AGENT_NAME=next_player
        )
    prompt = prompt.replace('\n', ' ')
    return prompt


def get_player_list(user_data, player):
    """
    Retrieves the list of players excluding the given player in the order they chat with the given player.
    """
    # Get all recipients for the player while maintaining order of first appearance
    players = user_data[user_data['sender_id'] == player]['recipient_id'].dropna()
    # Convert float values to int and then str, preserving order
    players = players.apply(lambda x: str(int(x)) if isinstance(x, float) else str(x))
    # Remove duplicates while preserving order
    return list(pd.unique(players))


def concat_with_pairs(user_data):
    """
    Concatenates consecutive messageSent entries and updates messageReceived entries.
    """
    num_rounds = int(user_data["chat_round_order"].max())
    
    for round in range(1, num_rounds + 1):
        pairs = get_pairs(user_data, round)
        for pair in pairs:
            sender, recipient = pair
            pair_mask = (
                ((user_data['sender_id'] == sender) & (user_data['recipient_id'] == recipient)) | 
                ((user_data['sender_id'] == recipient) & (user_data['recipient_id'] == sender))
            )
            pair_data = user_data[pair_mask]
            previous_sender = None
            previous_recipient = None

            last_recipient_event_order = 0
            last_write_event_order = 0

            for index, row in pair_data.iterrows():
                if row['event_type'] == 'message_sent':
                    if previous_sender == row['sender_id'] and previous_sender_recipient == row['recipient_id']:
                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'text'] += ' ' + row['text']
                        user_data.loc[(user_data['event_order'] == row['event_order']), 'text'] = ''

                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'time'] = row['time']
                        user_data.loc[(user_data['event_order'] == last_write_event_order), 'end_time'] = row['end_time']
                    else:
                        last_write_event_order = row['event_order']
                        previous_sender = row['sender_id']
                        previous_sender_recipient = row['recipient_id']
                elif row['event_type'] == 'message_recieved':
                    if previous_recipient == row['recipient_id'] and previous_recipient_sender == row['sender_id']:
                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'text'] += ' ' + row['text']
                        user_data.loc[(user_data['event_order'] == row['event_order']), 'text'] = ''

                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'time'] = row['time']
                        user_data.loc[(user_data['event_order'] == last_recipient_event_order), 'end_time'] = row['end_time']
                    else:
                        last_recipient_event_order = row['event_order']
                        previous_recipient = row['recipient_id']
                        previous_recipient_sender = row['sender_id']

    return user_data


def get_invalid_players(user_data):
    """
    Retrieves the invalid players from the user data.
    """
    # Get players without tweets in all 3 rounds
    players_with_tweets = user_data[user_data['event_type'] == 'tweet']['sender_id'].apply(lambda x: str(x)).value_counts()
    players_missing_tweets = players_with_tweets[players_with_tweets < 2].index.tolist()

    
    # Get players without exit survey
    all_players = user_data['sender_id'].dropna().apply(lambda x: str(x)).unique()
    players_with_survey = user_data[user_data['event_type'] == 'exit_survey']['worker_id'].dropna().apply(lambda x: str(x)).unique()
    players_missing_survey = [p for p in all_players if p not in players_with_survey]
    # Get players without any message_sent
    players_with_messages = user_data[user_data['event_type'] == 'message_sent']['sender_id'].dropna().apply(lambda x: str(x)).unique()
    players_missing_messages = [p for p in all_players if p not in players_with_messages]
    # Combine all invalid players
    invalid_players = list(set(players_missing_tweets + players_missing_survey + players_missing_messages))
    return invalid_players


def get_post_opinion(user_data, player, player_column_name):
    """
    Retrieves the initial opinion of a player from the user data.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        player (str): The name or identifier of the player.

    Returns:
        str: The initial opinion of the player.
    """
    opinion_slider_value_mapping = {
    "1": "Certainly disagree",
    "2": "Probably disagree",
    "3": "Lean disagree",
    "4": "Lean agree",
    "5": "Probably agree",
    "6": "Certainly agree",
    "unknown": "unknown"
}
    try:
        post_opinion = user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Post Opinion")]["text"].iloc[0]
    except:
        post_opinion = "unknown"
    try:
        slider_value = str(int(user_data[(user_data[player_column_name] == player) & (user_data["event_type"] == "Post Opinion")]["sliderValue"].iloc[0]))
    except:
        slider_value = "unknown"
    return post_opinion, opinion_slider_value_mapping[slider_value]


def get_post_opinion_text_prompt(topic):
    """
    Retrieves the post opinion text prompt.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "post_opinion_text.md")) as f:
        prompt_instructions = f.read()
        prompt = prompt_instructions.format(TOPIC=topic)
    return prompt


def get_post_opinion_likert_prompt(topic):
    """
    Retrieves the post opinion likert prompt.
    """
    with open(os.path.join(f"{BASE_DIR}/prompts", "fine_tune_v1", "post_opinion_likert.md")) as f:
        prompt_instructions = f.read()
        prompt = prompt_instructions.format(TOPIC=topic)
    return prompt


def format_data_for_chatgpt(
        user_data: pd.DataFrame, 
        player: str, 
        topic: str, 
        data_type: str, 
        filename: str,
        player_map: dict=None, 
        isAugmented=False, 
        isSplit=False, 
        validOnly=True,
        breadth=False,
        filter_type = "none",
        golden_filter = set()
        ):
    """
    Formats the data for a given player.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        player (str): The name or identifier of the player.
        topic (str): The topic of the conversation.
        isAugmented (bool): Whether the data is augmented.
    """
    invalid_players = get_invalid_players(user_data)

    dp = filename.replace(".csv", "")

    print(f"Invalid players: {invalid_players}")
    if player in invalid_players:
        print(f"[WARN] Player {player} is invalid, skip formatting")
        return []
    count = 0
    player_str = str(player)
    player_folder = f"{player_str}_augmented_v{count}" if isAugmented else player_str
    conversation_all_messages = []
    
    output_dir = os.path.join(DATA_DIR, INTERMEDIATE_DIR, data_type, dp, player_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # user_data = concat_with_pairs(user_data)

    all_players_except_current = get_player_list(user_data, player)
    chats_agent = {'messages': []}

    if not isSplit:
        demographic_backgrounds = get_demographic_background(user_data, "worker_id")
        initial_opinion_verbal, initial_opinion_likert = get_initial_opinion(user_data, player, "worker_id") # TODO: Update this to get the initial opinion on a Likert-scale
    conversation_history = get_conversation_history(user_data, player, all_players_except_current, all_messages=True)
    # Return early if conversation history is empty
    if not any(conversation_history[player].values()):
        print(f"[WARN] Player {player} has no conversation history, skip formatting")
        print(f"TOPIC: {topic}")
        return chats_agent

    isFirstMessage = True
    if not isSplit:
        chats_agent['messages'].append({
            'role': 'system',
            'content': get_system_prompt(demographic_backgrounds[player], topic, initial_opinion_likert, initial_opinion_verbal, player)
        })

    if filter_type in {"golden_players", "high_engage_golden_players"} and player not in golden_filter:
        print(f"[WARN] Current player {player} is not in golden, skip formatting")
        return []

    count = 0
    isFirstTweet = True
    for player2 in all_players_except_current:
        if player2 in invalid_players:
            print(f"Check {player}")
            continue
        if filter_type in {"golden_players", "high_engage_golden_players"} and player2 not in golden_filter:
            print(f"[Warn] p2 {player2} not in golden")
            continue
        if conversation_history[player][player2] == []:
            print(f"[WARN] Player {player} has no conversation history with {player2}, skip formatting")
            continue
        try:
            tweet_player, round_number = get_tweet(user_data, player, player2)
            tweet_player2, round_number = get_tweet(user_data, player2, player)
            player1_tweet = f"My Tweet: {tweet_player}"
            player2_tweet = f"{player2}'s Tweet: {tweet_player2}"

            if isFirstTweet:
                conversation_all_messages.append(player1_tweet)
                conversation_all_messages.append(player2_tweet)
                isFirstTweet = False
            else:
                generate_tweet_prompt = get_generate_tweet_prompt(player, player2, topic)
                conversation_all_messages.append(generate_tweet_prompt)
                messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                chats_agent['messages'].append({
                    'role': 'user',
                    'content': messages_till_now
                })
                chats_agent['messages'].append({
                    'role': 'assistant',
                    'content': player1_tweet  # "My Tweet: " is already in the message
                })
                with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                    f.write(json.dumps(chats_agent) + '\n')
                count += 1
                conversation_all_messages.pop() # Remove the prompt
                chats_agent['messages'].pop() # Remove the assistant message
                chats_agent['messages'].pop() # Remove the user message
                conversation_all_messages.append(player1_tweet)
                conversation_all_messages.append(player2_tweet)
        except IndexError as e:
            print(e)
            print(f"PLAYER: {player}, PLAYER2: {player2}, TOPIC: {topic}")
            print(f"Error occurred at line {e.__traceback__.tb_lineno}")
            continue

        for message in conversation_history[player][player2]:
            if message[0] == player:
                prompt = get_generate_message_prompt(player, player2, topic)
                messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                if message[2].lower() == "valid":
                    conversation_all_messages.append(prompt)
                    messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                    chats_agent['messages'].append({
                        'role': 'user',
                        'content': messages_till_now
                    })
                    chats_agent['messages'].append({
                        'role': 'assistant',
                        'content': f"My Response: {message[1]}"
                    })
                    with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                        f.write(json.dumps(chats_agent) + '\n')
                    count += 1
                    conversation_all_messages.pop() # Remove the prompt
                    chats_agent['messages'].pop() # Remove the assistant message
                    chats_agent['messages'].pop() # Remove the user message

                my_message = f"My Response: {message[1]}"
                conversation_all_messages.append(my_message) # Add the message to the conversation for next round
            else:
                their_message = f"{player2}'s Response: {message[1]}"
                conversation_all_messages.append(their_message)
        
        if all_players_except_current.index(player2) < len(all_players_except_current) - 1:
            next_player = all_players_except_current[all_players_except_current.index(player2) + 1]
            if next_player in invalid_players:
                continue
            conversation_all_messages.append(get_change_round_prompt(player2, next_player))

    post_opinion_text_prompt = get_post_opinion_text_prompt(topic)
    post_opinion_likert_prompt = get_post_opinion_likert_prompt(topic)
    post_opinion_text, post_opinion_likert = get_post_opinion(user_data, player, "worker_id")
    allow_post_opinion = is_post_opinion_valid_rejudge(dp, player)

    if post_opinion_text != "unknown" and allow_post_opinion:
        conversation_all_messages.append(post_opinion_text_prompt)
        messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
        chats_agent['messages'].append({
            'role': 'user',
            'content': messages_till_now
        })
        chats_agent['messages'].append({
            'role': 'assistant',
            'content': f"{post_opinion_text}"
        })
        with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
            f.write(json.dumps(chats_agent) + '\n')
        count += 1
        conversation_all_messages.pop() # Remove the prompt
        chats_agent['messages'].pop() # Remove the assistant message
        chats_agent['messages'].pop() # Remove the user message
        my_message = f"{post_opinion_text}"
        conversation_all_messages.append(my_message) # Add the message to the conversation for next round

    if post_opinion_likert != "unknown" and allow_post_opinion:
        conversation_all_messages.append(post_opinion_likert_prompt)
        messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
        chats_agent['messages'].append({
            'role': 'user',
            'content': messages_till_now
        })
        chats_agent['messages'].append({
            'role': 'assistant',
            'content': f"{post_opinion_likert}"
        })
        with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
            f.write(json.dumps(chats_agent) + '\n')
        count += 1
        conversation_all_messages.pop() # Remove the prompt
        chats_agent['messages'].pop() # Remove the assistant message
        chats_agent['messages'].pop() # Remove the user message
        my_message = f"{post_opinion_likert}"
        conversation_all_messages.append(my_message) # Add the message to the conversation for next round

    return chats_agent


def format_data_for_chatgpt_round_split(
        user_data: pd.DataFrame, 
        player: str, 
        topic: str, 
        data_type: str, 
        player_map: dict=None,
        isAugmented=False, 
        isSplit=False, 
        validOnly=True, 
        set_type: str="test",
        breadth=False,
        filter_type = "none",
        golden_filter=set()
        ):
    """
    Formats the data for a given player.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data.
        player (str): The name or identifier of the player.
        topic (str): The topic of the conversation.
        isAugmented (bool): Whether the data is augmented.
    """
    invalid_players = get_invalid_players(user_data)
    print(f"Invalid players: {invalid_players}")
    if player in invalid_players:
        print(f"[WARN] Player {player} is invalid, skip formatting")
        return []
    count = 0
    player_str = str(player)
    player_folder = f"{player_str}_augmented_v{count}" if isAugmented else player_str
    conversation_all_messages = []
    
    output_dir = os.path.join(DATA_DIR, "chatgpt_data", data_type, player_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if filter_type in {"golden_players", "high_engage_golden_players"} and player not in golden_filter:
        print(f"[WARN] Current player {player} is not in golden, skip formatting")
        return []
    

    # user_data = concat_with_pairs(user_data)

    all_players_except_current = get_player_list(user_data, player)
    chats_agent = {'messages': []}

    if not isSplit:
        demographic_backgrounds = get_demographic_background(user_data, "worker_id")
        initial_opinion_verbal, initial_opinion_likert = get_initial_opinion(user_data, player, "worker_id") # TODO: Update this to get the initial opinion on a Likert-scale
    conversation_history = get_conversation_history(user_data, player, all_players_except_current, all_messages=True)
    # Return early if conversation history is empty
    if not any(conversation_history[player].values()):
        print(f"[WARN] Player {player} has no conversation history, skip formatting")
        print(f"TOPIC: {topic}")
        return chats_agent

    isFirstMessage = True
    if not isSplit:
        chats_agent['messages'].append({
            'role': 'system',
            'content': get_system_prompt(demographic_backgrounds[player], topic, initial_opinion_likert, initial_opinion_verbal, player)
        })

    count = 0
    isFirstTweet = True
    count_round_3 = 0
    for player2 in all_players_except_current:
        if player2 in invalid_players:
            print(f"Check {player}")
            continue
        if filter_type in {"golden_players", "high_engage_golden_players"} and player2 not in golden_filter:
            print(f"[Warn] p2 {player2} not in golden")
            continue

        if conversation_history[player][player2] == []:
            print(f"[WARN] Player {player} has no conversation history with {player2}, skip formatting")
            continue
        try:
            tweet_player, round_number = get_tweet(user_data, player, player2)
            tweet_player2, round_number = get_tweet(user_data, player2, player)
            if set_type == "train" and round_number == 3:
                continue

            player1_tweet = f"My Tweet: {tweet_player}"
            player2_tweet = f"{player2}'s Tweet: {tweet_player2}"

            if isFirstTweet:
                conversation_all_messages.append(player1_tweet)
                conversation_all_messages.append(player2_tweet)
                isFirstTweet = False
            else:
                if set_type == "test" and int(round_number) == 3:
                    generate_tweet_prompt = get_generate_tweet_prompt(player, player2, topic)
                    conversation_all_messages.append(generate_tweet_prompt)
                    messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                    chats_agent['messages'].append({
                        'role': 'user',
                        'content': messages_till_now
                    })
                    chats_agent['messages'].append({
                        'role': 'assistant',
                        'content': f"{player1_tweet}"
                    })
                    with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                        f.write(json.dumps(chats_agent) + '\n')
                    count += 1
                    conversation_all_messages.pop() # Remove the prompt
                    chats_agent['messages'].pop() # Remove the assistant message
                    chats_agent['messages'].pop() # Remove the user message
                elif set_type == "train" and int(round_number) != 3:
                    generate_tweet_prompt = get_generate_tweet_prompt(player, player2, topic)
                    conversation_all_messages.append(generate_tweet_prompt)
                    messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                    chats_agent['messages'].append({
                        'role': 'user',
                        'content': messages_till_now
                    })
                    chats_agent['messages'].append({
                        'role': 'assistant',
                        'content': f"{player1_tweet}"
                    })
                    with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                        f.write(json.dumps(chats_agent) + '\n')
                    count += 1
                    conversation_all_messages.pop() # Remove the prompt
                    chats_agent['messages'].pop() # Remove the assistant message
                    chats_agent['messages'].pop() # Remove the user message

                conversation_all_messages.append(player1_tweet)
                conversation_all_messages.append(player2_tweet)
        except IndexError as e:
            print(e)
            print(f"PLAYER: {player}, PLAYER2: {player2}, TOPIC: {topic}")
            print(f"Error occurred at line {e.__traceback__.tb_lineno}")
            continue

        for message in conversation_history[player][player2]:
            if message[0] == player:
                prompt = get_generate_message_prompt(player, player2, topic)
                if message[2].lower() == "valid":
                    if set_type == "test" and int(message[3]) == 3:
                        conversation_all_messages.append(prompt)
                        messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                        count_round_3 += 1
                        chats_agent['messages'].append({
                            'role': 'user',
                            'content': messages_till_now
                        })
                        chats_agent['messages'].append({
                            'role': 'assistant',
                            'content': f"My Response: {message[1]}"
                        })
                        with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                            f.write(json.dumps(chats_agent) + '\n')
                        count += 1
                        conversation_all_messages.pop() # Remove the prompt
                        chats_agent['messages'].pop() # Remove the assistant message
                        chats_agent['messages'].pop() # Remove the user message

                    elif set_type == "train" and int(message[3]) != 3:
                        conversation_all_messages.append(prompt)
                        messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
                        chats_agent['messages'].append({
                            'role': 'user',
                            'content': messages_till_now
                        })
                        chats_agent['messages'].append({
                            'role': 'assistant',
                            'content': f"My Response: {message[1]}"
                        })
                        with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                            f.write(json.dumps(chats_agent) + '\n')
                        count += 1
                        conversation_all_messages.pop() # Remove the prompt
                        chats_agent['messages'].pop() # Remove the assistant message
                        chats_agent['messages'].pop() # Remove the user message
                my_message = f"My Response: {message[1]}"
                conversation_all_messages.append(my_message) # Add the message to the conversation for next round
            else:
                their_message = f"{player2}'s Response: {message[1]}"
                conversation_all_messages.append(their_message)
        
        if all_players_except_current.index(player2) < len(all_players_except_current) - 1:
            next_player = all_players_except_current[all_players_except_current.index(player2) + 1]
            if next_player in invalid_players:
                continue
            conversation_all_messages.append(get_change_round_prompt(player2, next_player))

    if set_type == 'test':
        post_opinion_text_prompt = get_post_opinion_text_prompt(topic)
        post_opinion_likert_prompt = get_post_opinion_likert_prompt(topic)
        post_opinion_text, post_opinion_likert = get_post_opinion(user_data, player, "worker_id")
        if post_opinion_text != "unknown":
            conversation_all_messages.append(post_opinion_text_prompt)
            messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
            chats_agent['messages'].append({
                'role': 'user',
                'content': messages_till_now
            })
            chats_agent['messages'].append({
                'role': 'assistant',
                'content': f"{post_opinion_text}"
            })
            with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                f.write(json.dumps(chats_agent) + '\n')
            count += 1
            conversation_all_messages.pop() # Remove the prompt
            chats_agent['messages'].pop() # Remove the assistant message
            chats_agent['messages'].pop() # Remove the user message
            my_message = f"{post_opinion_text}"
            conversation_all_messages.append(my_message) # Add the message to the conversation for next round

        if post_opinion_likert != "unknown":
            conversation_all_messages.append(post_opinion_likert_prompt)
            messages_till_now = '\n'.join(conversation_all_messages) # Join the conversation history and the prompt for the user message
            chats_agent['messages'].append({
                'role': 'user',
                'content': messages_till_now
            })
            chats_agent['messages'].append({
                'role': 'assistant',
                'content': f"{post_opinion_likert}"
            })
            with open(os.path.join(output_dir, f"message_{count}.jsonl"), "w") as f:
                f.write(json.dumps(chats_agent) + '\n')
            count += 1
            conversation_all_messages.pop() # Remove the prompt
            chats_agent['messages'].pop() # Remove the assistant message
            chats_agent['messages'].pop() # Remove the user message
            my_message = f"{post_opinion_likert}"
            conversation_all_messages.append(my_message) # Add the message to the conversation for next round

    return chats_agent


def format_data_for_model(
        user_data, 
        topic, 
        data_type, 
        player_map,
        filename,
        isAugmented=False, 
        isSplit=False, 
        split_type=None, 
        set_type=None,
        breadth=False,
        filter_type="none",
        golden_filter=set(),
        ):
    players = user_data["worker_id"].dropna().unique()
    for player in players:
        if split_type == "round":
            format_data_for_chatgpt_round_split(
                user_data, 
                player, 
                topic, 
                data_type, 
                player_map, 
                isAugmented, 
                isSplit, 
                validOnly=True, 
                set_type=set_type,
                breadth=breadth,
                filter_type = filter_type,
                golden_filter=golden_filter
                )
        else:
            format_data_for_chatgpt(
                user_data, 
                player, 
                topic, 
                data_type, 
                filename=filename,
                player_map=player_map,
                isAugmented=isAugmented,
                isSplit=isSplit,
                validOnly=True,
                breadth=breadth,
                filter_type = filter_type,
                golden_filter=golden_filter
                )


def map_players_to_ids(user_data):
    """
    Maps empirica player IDs to worker IDs.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data with empirica_id and worker_id columns

    Returns:
        dict: A dictionary mapping empirica_ids to worker_ids
    """
    players = user_data["empirica_id"].dropna().unique()
    player_id_map = {}
    for player in players:
        worker_id = user_data[user_data["empirica_id"] == player]["worker_id"].iloc[0]
        player_id_map[player] = worker_id
    return player_id_map


def extract_topic_from_filename(filename: str) -> str:
    """
    Extracts the topic from a filename.
    
    Args:
        filename (str): The full filename containing the topic
        
    Returns:
        str: The cleaned topic string
    """
    topic = re.search(r'\d{8}_\d{6}_(.*)_.{26}', filename).group(1).replace('_', ' ')
    topic = re.sub(r' +', ' ', topic)

    return topic


def correct_data_types(user_data):
    """
    Corrects data types for ID columns in the DataFrame.

    Args:
        user_data (pd.DataFrame): The DataFrame containing user data

    Returns:
        pd.DataFrame: DataFrame with corrected data types
    """
    # Convert ID columns to string type
    id_columns = ['worker_id', 'sender_id', 'recipient_id']
    for col in id_columns:
        if col in user_data.columns:
            user_data[col] = user_data[col].apply(lambda x: str(int(x)) if pd.notna(x) else x)
    
    return user_data

