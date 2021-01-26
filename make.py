#!/usr/bin/env python3

import json

from parlai.core.agents import create_agent
from parlai.scripts.script import ParlaiScript
from parlai.core.params import ParlaiParser
from parlai.utils.strings import normalize_reply
import parlai.utils.logging as logging
import copy
import os
import sys
from tqdm import tqdm

SILENCE = '__SILENCE__'
PERSONA_PREFIX = "your persona: "


def setup_args():
    parser = ParlaiParser(True, True)
    return parser


def _run_conversation(conversation_id, conversation, agent, agent2):
    agent.reset()
    agent2.reset()
    model_persona = conversation['model_persona']
    model_persona = PERSONA_PREFIX + model_persona.replace("\n", "\n" + PERSONA_PREFIX)
    human_persona = conversation['human_persona']
    human_persona = PERSONA_PREFIX + human_persona.replace("\n", "\n" + PERSONA_PREFIX)

    text2 = SILENCE

    dialogue = []

    for turn_id in range(6):

        prefix = ""
        if turn_id == 0:
            prefix = model_persona + '\n'
        obs1 = prefix + text2
        observed = agent.observe(
            {'id': 'SPEAKER_1', 'text': obs1, 'episode_done': False}
        )
        response = agent.act()
        text1 = normalize_reply(response['text']).strip()
        if turn_id == 0:
            prefix = human_persona + '\n'
        obs2 = prefix + text1
        observed = agent2.observe(
            {'id': 'SPEAKER_2', 'text': obs2, 'episode_done': False}
        )
        response = agent2.act()
        text2 = normalize_reply(response['text']).strip()

        dialogue.append([text1, text2])

    return dialogue


def singlerun(parser, opt):
    agent = create_agent(opt, True)
    agent2 = create_agent(opt, True)
    parser.opt = agent.opt

    all_convs = []

    with open(opt['eval_file']) as eval_file:
        lines = []
        for i, line in enumerate(eval_file):
            lines.append(line)
        for i in tqdm(range(len(lines))):
            line = lines[i]
            if not line.strip():
                continue
            conversation = json.loads(line)
            conv = _run_conversation(i, conversation, agent, agent2)
            all_convs.append(conv)
    return all_convs


if __name__ == '__main__':
    
    import model_configs
    print(sys.argv)
    for evalfile in os.listdir('evallog'):
        agent_name = evalfile.split('.')[0]
        if 'human' in agent_name:
            continue
        config = getattr(model_configs, agent_name)

        parser = setup_args()
        opt = parser.parse_args()
        opt.update(config)
        opt['override'] = config 
        print("==================== Evaluating Agent", agent_name)
        opt.update({'eval_file': 'evallog/'+evalfile})
        #import ipdb; ipdb.set_trace()
        all_convs = singlerun(parser, opt)
        import json
        json.dump(all_convs, open('selfplay/'+evalfile,'w'))














