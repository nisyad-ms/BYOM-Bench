"""
Data preparation functions for PersonaMem.

Core data generation functions for the PersonaMem pipeline.
Uses relative imports from this package and structured outputs for conversation generation.

Key functions:
    - prepare_persona: Load/create persona and expand it
    - prepare_topics: Set up topic-specific source data
    - prepare_data_on_other_topics: Generate conversation data for most topics
    - prepare_data_on_writing_topic: Generate conversation data for writing/email/coding topics
    
Note: This module uses OpenAI Structured Outputs for conversation generation,
ensuring type-safe, validated output without manual JSON parsing.
"""

import json
import os
import random
import re

from json_repair import (
    repair_json,  # Still needed for history parsing (non-conversation data)
)
from tqdm import tqdm

# Use relative imports within the package
from . import prompts, utils
from .query_llm import QueryLLM
from .schemas import GeneratedConversation


def prepare_persona(LLM, idx_persona, all_personas, args):
    # Load a random persona
    found = utils.find_existing_persona_files(idx_persona)
    # found = False
    if found:
        # Ensure that every data file with the same idx_persona share the same persona
        persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year \
            = found['persona'], found['expanded_persona'], found['start_time'], found['init_general_personal_history'], found['general_personal_history_next_week'], found['general_personal_history_next_month'], found['general_personal_history_next_year']
        LLM.expanded_persona = expanded_persona
        if not start_time:
            start_time = utils.pick_a_random_time()
        if args['inference']['verbose']:
            print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}')
            print(persona)
            print(f'{utils.Colors.OKGREEN}{"Expanded Persona"}:{utils.Colors.ENDC}')
            print(expanded_persona)
    else:
        # Create a new persona for the new idx_persona
        random_row = random.choice(all_personas)
        persona = random_row.strip()[13:-2]  # Remove prefix '{"persona":' and suffix '"}'
        if args['inference']['verbose']:
            print(f'{utils.Colors.OKGREEN}{"Original Persona"}:{utils.Colors.ENDC}{persona}')

        # Expand the persona to at least five sentences
        start_time = utils.pick_a_random_time()
        expanded_persona = LLM.query_llm(step='expand_persona', persona=persona, start_time=start_time, verbose=args['inference']['verbose'])
        init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year = None, None, None, None

    return persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year


def prepare_topics(idx_topic, all_topics, curr_topic, args):
    # Process each topic as needed
    print(f'{utils.Colors.OKBLUE}Processing topic: {curr_topic}, {idx_topic + 1}/{len(all_topics)}{utils.Colors.ENDC}')

    # Load a random conversation history from the chosen real-world dataset
    if curr_topic == 'writing':
        source_dir = args['datasets']['writing_source_dir']
    elif curr_topic == 'email':
        source_dir = args['datasets']['email_source_dir']
    elif curr_topic == 'coding':
        source_dir = args['datasets']['coding_source_dir']
    elif curr_topic == 'legal':
        source_dir = args['datasets']['legal_source_dir']
    elif curr_topic == 'therapy':
        source_dir = args['datasets']['therapy_source_dir']
    else:
        source_dir = None

    all_source_files = utils.load_all_source_data(source_dir, curr_topic) if source_dir is not None else None
    return source_dir, all_source_files


def generate_conversation_structured(
    LLM,
    topic: str,
    period: str,
    verbose: bool = False
) -> GeneratedConversation:
    """Generate a conversation using structured outputs.

    Uses OpenAI's structured output API to guarantee valid GeneratedConversation
    objects, eliminating the need for regex parsing and JSON repair.

    Args:
        LLM: The QueryLLM instance
        topic: Conversation topic (travel, therapy, food, etc.)
        period: Time period (INIT, WEEK, MONTH, YEAR)
        verbose: Whether to print verbose output

    Returns:
        GeneratedConversation: Validated conversation object
    """
    # Get the appropriate personal history based on period
    period_to_history = {
        'INIT': 'init_personal_history',
        'WEEK': 'first_expand_personal_history',
        'MONTH': 'second_expand_personal_history',
        'YEAR': 'third_expand_personal_history'
    }
    curr_personal_history = getattr(LLM, period_to_history[period])

    # Build the prompt
    prompt = prompts.prompts_for_generating_conversations(
        topic,
        LLM.expanded_persona,
        curr_personal_history=curr_personal_history,
        period=period
    )

    # Query with structured output - guaranteed valid schema
    conversation = LLM.query_llm_structured(
        prompt=prompt,
        response_schema=GeneratedConversation,
        verbose=verbose
    )

    if verbose:
        print(f'{utils.Colors.OKGREEN}Generated conversation with {len(conversation.turns)} turns{utils.Colors.ENDC}')

    return conversation


def prepare_data_on_writing_topic(LLM, topic, persona, source_data, output_file_path, args, data_collector: dict = None):
    # Convert the writing sample into a conversation
    preferences = LLM.query_llm(step='prepare_new_content', data=persona, action='preferences', data_type=topic, verbose=args['inference']['verbose'])
    if topic == 'coding':
        source_data = LLM.query_llm(step='translate_code', persona=preferences, data=source_data, verbose=args['inference']['verbose'])
    elif topic == 'email':
        source_data = LLM.query_llm(step='rewrite_email', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])
    elif topic == 'writing':
        source_data = LLM.query_llm(step='rewrite_creative_writing', persona={'persona': persona, 'preferences': preferences}, data=source_data, verbose=args['inference']['verbose'])

    updated_writing_sample = LLM.query_llm(step='prepare_new_content', data=source_data, action='rewrite_from_persona', data_type=topic, verbose=args['inference']['verbose'])
    if 'python' in preferences or 'plaintext' in preferences:
        preferences = preferences.strip("```python").strip("```plaintext").strip()
    if 'plaintext' in updated_writing_sample:
        updated_writing_sample = updated_writing_sample.strip("```plaintext").strip()

    conversation = LLM.query_llm(step='prepare_new_content', action='rewrite_as_conversation', data_type=topic, verbose=args['inference']['verbose'])
    if conversation.startswith('```python'):
        conversation = conversation.replace('```python', '', 1)
    conversation = conversation.strip("```plaintext")
    try:
        conversation = json.loads(conversation)
    except:
        conversation = conversation

    responses = [source_data, preferences, updated_writing_sample, conversation]
    if topic == 'coding':
        data_names = ['Original Sample', 'Coding and Formatting Styles', 'Updated Coding Sample', 'Conversation']
    else:
        data_names = ['Original Sample', 'Writing and Formatting Styles', 'Updated Writing Sample', 'Conversation']
    for response, data_name in zip(responses, data_names):
        if data_collector is not None:
            data_collector[data_name] = utils.extract_json_from_response(response, parse_json=False, parse_list=False)
        else:
            utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)

    return data_collector


def prepare_data_on_other_topics(LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                 init_general_personal_history, first_expand_general_personal_history, second_expand_general_personal_history, third_expand_general_personal_history, args,
                                 data_collector: dict = None):
    # Feed the thread with a seeding data from the real-world conversation
    if source_dir is not None:
        source_conversation = utils.preprocess_source_data(source_data, curr_topic)
        _ = LLM.query_llm(step='source_data', seed=source_conversation, verbose=args['inference']['verbose'])
    else:
        _ = LLM.query_llm(step='elaborate_topic', topic=curr_topic, verbose=args['inference']['verbose'])

    # Check for quick/debug modes
    quick_mode = args['inference'].get('quick_mode', False)
    debug_mode = args['inference'].get('debug_mode', False)

    if debug_mode:
        # DEBUG MODE: Absolute minimum for testing core logic (~30 seconds)
        # Only init steps, no expansion - tests all code paths with minimal LLM calls
        print(f'{utils.Colors.WARNING}DEBUG MODE: Init steps only, no expansion (testing core logic){utils.Colors.ENDC}')
        steps = ['init_general_personal_history', 'init_contextual_personal_history']
        data_names = ['Init General Personal History', 'Init Contextual Personal History']
    elif quick_mode:
        # QUICK MODE: Init + one expansion (to get preference updates for TOD evaluation)
        print(f'{utils.Colors.WARNING}QUICK MODE: Generating init + next_week steps (for preference evolution){utils.Colors.ENDC}')
        steps = ['init_general_personal_history', 'first_expand_general_personal_history',
                 'init_contextual_personal_history', 'first_expand_contextual_personal_history']
        data_names = ['Init General Personal History', 'General Personal History Next Week',
                      'Init Contextual Personal History', 'Contextual Personal History Next Week']
    else:
        # FULL MODE: All 8 history steps
        steps = ['init_general_personal_history', 'first_expand_general_personal_history', 'second_expand_general_personal_history', 'third_expand_general_personal_history',
                 'init_contextual_personal_history', 'first_expand_contextual_personal_history', 'second_expand_contextual_personal_history', 'third_expand_contextual_personal_history']
        data_names = ['Init General Personal History', 'General Personal History Next Week', 'General Personal History Next Month', 'General Personal History Next Year',
                      'Init Contextual Personal History', 'Contextual Personal History Next Week', 'Contextual Personal History Next Month', 'Contextual Personal History Next Year']

    existing_general_personal_history = {'init_general_personal_history': init_general_personal_history, 'first_expand_general_personal_history': first_expand_general_personal_history,
                                         'second_expand_general_personal_history': second_expand_general_personal_history, 'third_expand_general_personal_history': third_expand_general_personal_history}

    last_timestamps = []
    for step, data_name in tqdm(zip(steps, data_names)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')
        if step in existing_general_personal_history:
            if existing_general_personal_history[step] is not None:
                if step == 'init_general_personal_history':
                    print('Loading existing general personal history.')
                # Use existing general personal history shared across multiple topics for the same persona
                if data_collector is not None:
                    data_collector[data_name] = utils.extract_json_from_response('```json' + str(existing_general_personal_history[step]) + '```', parse_json=True, parse_list=False)
                else:
                    utils.append_json_to_file('```json' + str(existing_general_personal_history[step]) + '```', output_file_path, curr_data_name=data_name, parse_json=True)
                continue
            else:
                if step == 'init_general_personal_history':
                    print('Generating new general personal history.')

        response = LLM.query_llm(step=step, persona=expanded_persona, topic=curr_topic, idx_topic=idx_topic, start_time=start_time, verbose=args['inference']['verbose'])

        if step == 'init_contextual_personal_history':
            text_before_json = re.split(r'```json', response)[0].strip()
            if data_collector is not None:
                data_collector["Topic-Specific Hobbies"] = utils.extract_json_from_response(text_before_json, parse_json=False, parse_list=False)
            else:
                utils.append_json_to_file(text_before_json, output_file_path, curr_data_name="Topic-Specific Hobbies", parse_json=False)
            try:
                json_part = re.split(r'```json', response)[1].strip()
            except:
                json_part = response
            json_part = repair_json('[{'+json_part+'}]')
            json_part = utils.filter_valid_dates(json_part)
            if data_collector is not None:
                data_collector[data_name] = utils.extract_json_from_response(json_part, parse_json=False, parse_list=False)
            else:
                utils.append_json_to_file(json_part, output_file_path, curr_data_name=data_name, parse_json=False)
            last_timestamps.append(utils.extract_last_timestamp(json_part))
        else:
            response = repair_json('[{'+response+'}]')
            response = utils.filter_valid_dates(response)
            if data_collector is not None:
                data_collector[data_name] = utils.extract_json_from_response(response, parse_json=False, parse_list=False)
            else:
                utils.append_json_to_file(response, output_file_path, curr_data_name=data_name, parse_json=False)
            last_timestamps.append(utils.extract_last_timestamp(response))

    # Populate personal history into conversation
    if debug_mode:
        # DEBUG MODE: 1 conversation only (fastest - for testing core logic)
        conv_steps = ['init_conversation']
        conv_data_names = ['Init Conversation']
        conv_periods = ['INIT']
    elif quick_mode:
        # QUICK MODE: 2 conversations (init + next_week to include preference updates)
        conv_steps = ['init_conversation', 'first_expand_conversation']
        conv_data_names = ['Init Conversation', 'Conversation Next Week']
        conv_periods = ['INIT', 'WEEK']
    else:
        # FULL MODE: All 4 conversation periods
        conv_steps = ['init_conversation', 'first_expand_conversation', 'second_expand_conversation', 'third_expand_conversation']
        conv_data_names = ['Init Conversation', 'Conversation Next Week', 'Conversation Next Month', 'Conversation Next Year']
        conv_periods = ['INIT', 'WEEK', 'MONTH', 'YEAR']

    last_timestamps = utils.merge_timestamps(last_timestamps)

    # Generate conversations using structured outputs (all modes)
    for conv_idx, (step, data_name, period) in enumerate(zip(conv_steps, conv_data_names, conv_periods)):
        print(f'{utils.Colors.OKGREEN}Processing step: {step}{utils.Colors.ENDC}')

        # Use structured output for conversation generation
        conversation = generate_conversation_structured(
            LLM=LLM,
            topic=curr_topic,
            period=period,
            verbose=args['inference']['verbose']
        )

        print(f'{utils.Colors.OKGREEN}Generated conversation with {len(conversation.turns)} turns for {period}{utils.Colors.ENDC}')

        # Store the GeneratedConversation object directly
        if data_collector is not None:
            data_collector[data_name] = conversation
        else:
            # For file output, serialize to JSON
            utils.append_json_to_file(
                conversation.model_dump_json(indent=2),
                output_file_path,
                curr_data_name=data_name,
                parse_json=True
            )

    return data_collector


def prepare_irrelevant_contexts(LLM, args):
    with open(args['datasets']['random_questions_file'], 'r') as file:
        all_random_questions = [line.strip() for line in file]
    with open(args['datasets']['random_code_questions_file'], 'r') as file:
        all_random_code_questions = [line.strip() for line in file]
    all_random_questions = all_random_questions + all_random_code_questions

    output_file_path = os.path.join(args['inference']['output_dir'], 'irrelevant_contexts.json')
    for index, question in enumerate(tqdm(all_random_questions)):
        LLM.create_a_thread(step='irrelevant')

        model_answer = LLM.query_llm(step='random_question', data=question, verbose=args['inference']['verbose'])
        follow_up_question = LLM.query_llm(step='random_question_follow_up', verbose=args['inference']['verbose'])
        follow_up_answer = LLM.query_llm(step='random_question_follow_up_response', data=follow_up_question, verbose=args['inference']['verbose'])

        new_entry = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": model_answer},
            {"role": "user", "content": follow_up_question},
            {"role": "assistant", "content": follow_up_answer}
        ]

        LLM.delete_a_thread(step='irrelevant')

        existing_data = []
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r", encoding="utf-8") as file:
                    existing_data = json.load(file)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except json.JSONDecodeError:
                existing_data = []

        existing_data.append({str(index): new_entry})
        with open(output_file_path, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)


def prepare_data(args):
    # Load all personas
    with open(args['datasets']['persona_file'], 'r') as file:
        all_personas = file.readlines()

    if args['datasets']['topics'] == ['irrelevant']:
        LLM = QueryLLM(args)
        prepare_irrelevant_contexts(LLM, args)
    else:
        # Generate conversational data relevant to the topic and the persona
        all_errored_data_paths = {}

        for idx_persona in tqdm(range(int(args['inference']['start_persona_idx']), int(args['inference']['num_personas']))):
            LLM = QueryLLM(args)
            persona, expanded_persona, start_time, init_general_personal_history, general_personal_history_next_week, \
                general_personal_history_next_month, general_personal_history_next_year = prepare_persona(LLM, idx_persona, all_personas, args)

            # Clean up the names of topics
            if args['datasets']['topics'] == ['all']:
                all_topics = utils.get_all_topic_names()
            else:
                all_topics = [ctx.strip() for ctx in args['datasets']['topics']]

            # Since we assign a consecutive time frame for all topics, we randomly permute topics to ensure generalization
            if len(all_topics) > 1:
                random.shuffle(all_topics)

                # Ensure "coding," "writing," or "email" is not the first topic
                restricted_topics = {"coding", "writing", "email"}
                if all_topics[0] in restricted_topics:
                    for i in range(1, len(all_topics)):
                        if all_topics[i] not in restricted_topics:
                            all_topics[0], all_topics[i] = all_topics[i], all_topics[0]
                            break

            # Loop through each topic in the list
            for idx_topic, curr_topic in tqdm(enumerate(all_topics)):
                if curr_topic == '' or curr_topic is None:
                    continue
                source_dir, all_source_files = prepare_topics(idx_topic, all_topics, curr_topic, args)

                # Set a consecutive time frame for different topics for each persona, while all samples below are independent
                if idx_topic > 0:
                    start_time = utils.pick_a_random_time_within_a_year(start_time)

                for idx_sample in range(int(args['inference']['start_sample_idx']), int(args['inference']['num_samples_per_topic'])):
                    LLM = QueryLLM(args)

                    output_file_path = os.path.join(args['inference']['output_dir'],
                                                    os.path.join(f'{curr_topic}', f'{args["inference"]["output_file_name"]}_{curr_topic}_persona{idx_persona}_sample{idx_sample}.json'))
                    utils.append_json_to_file(persona, output_file_path, curr_data_name='Original Persona', parse_json=False)
                    utils.append_json_to_file(expanded_persona, output_file_path, curr_data_name='Expanded Persona', parse_json=False)
                    utils.append_json_to_file(curr_topic, output_file_path, curr_data_name='Topic', parse_json=False)
                    print(f'{utils.Colors.OKGREEN}Output file path: {output_file_path}{utils.Colors.ENDC}')

                    # Load a random source data to the LLM as a background memory about the topic
                    source_data = utils.load_one_source_data(source_dir, all_source_files, curr_topic) if all_source_files is not None else None
                    try:
                        if curr_topic == 'writing' or curr_topic == 'email' or curr_topic == 'coding':
                            LLM.create_a_thread(step='writing')
                            prepare_data_on_writing_topic(LLM, curr_topic, persona, source_data, output_file_path, args)
                        else:
                            LLM.create_a_thread(step='conversation')
                            prepare_data_on_other_topics(LLM, expanded_persona, source_data, source_dir, curr_topic, idx_topic, start_time, output_file_path,
                                                         init_general_personal_history, general_personal_history_next_week, general_personal_history_next_month, general_personal_history_next_year, args)
                    except Exception as e:
                        print(f'{utils.Colors.FAIL}Error at generating file{output_file_path}: {e}{utils.Colors.ENDC}')
                        all_errored_data_paths[output_file_path] = e

        if len(all_errored_data_paths) > 0:
            print(f'{utils.Colors.FAIL}All errored data paths: {utils.Colors.ENDC}')
            for key, value in all_errored_data_paths.items():
                print(key)
        else:
            print(f'{utils.Colors.OKGREEN}All data are successfully generated.{utils.Colors.ENDC}')
