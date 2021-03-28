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


def _run_conversation(conversation_id, conversation, agent):
    agent.reset()
    model_persona = conversation['model_persona']
    model_persona = PERSONA_PREFIX + model_persona.replace("\n", "\n" + PERSONA_PREFIX)

    ori_dialog = conversation['dialog']
    text2 = SILENCE
    prefix = model_persona + '\n'

    refs = []
    gens = []

    for turn in ori_dialog:
        if turn['speaker'] == 'human_evaluator':
            text2 = turn['text']
            obs1 = prefix + text2
            prefix = ""
            observed = agent.observe(
                {'id': 'SPEAKER_1', 'text': obs1, 'episode_done': False}
            )
        if turn['speaker'] == 'model':
            refs.append(turn['text'])
            if prefix != "":
                obs1 = prefix + text2
                prefix = ""
                observed = agent.observe(
                    {'id': 'SPEAKER_1', 'text': obs1, 'episode_done': False}
                )
            response = agent.act()
            text1 = normalize_reply(response['text']).strip()
            gens.append(text1)

    assert len(refs) == len(gens)
    return refs, gens


def singlerun(parser, opt):
    agent = create_agent(opt, True)
    parser.opt = agent.opt

    refs = []
    gens = []

    with open('evallog/human_eval.jsonl') as eval_file:
        lines = []
        for i, line in enumerate(eval_file):
            lines.append(line)
        for i in tqdm(range(len(lines))):
            line = lines[i]
            if not line.strip():
                continue
            conversation = json.loads(line)
            _refs, _gens = _run_conversation(i, conversation, agent)
            refs = refs+_refs
            gens = gens+_gens
    return refs, gens


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
        refs, gens = singlerun(parser, opt)
        with open('static_eval/'+evalfile.replace('.jsonl','')+"_ref.txt",'w') as fout:
            for l in refs:
                fout.write(l+'\n')
        with open('static_eval/'+evalfile.replace('.jsonl','')+"_gen.txt",'w') as fout:
            for l in gens:
                fout.write(l+'\n')














