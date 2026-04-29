import json
from torch.utils.data import Dataset


class WebQSPDataset(Dataset):
    """Base WebQSP dataset. Subclass to add baseline-specific fields."""

    def __init__(self, data_path):
        with open(data_path) as f:
            self.samples = [self._parse(q) for q in json.load(f)['Questions']]

    def _parse(self, q):
        parses = q['Parses']
        main_parse = next((p for p in parses if p['Answers']), parses[0])

        seen, answers = set(), []
        for p in parses:
            for a in p['Answers']:
                if a['AnswerArgument'] not in seen:
                    seen.add(a['AnswerArgument'])
                    answers.append({
                        'mid': a['AnswerArgument'],
                        'name': a.get('EntityName') or a['AnswerArgument'],
                    })

        return {
            'id': q['QuestionId'],
            'question': q['ProcessedQuestion'],
            'topic_entity': {
                'mid': main_parse['TopicEntityMid'],
                'name': main_parse['TopicEntityName'],
            },
            'answers': answers,
            'relation_path': main_parse.get('InferentialChain') or [],
            'sparql': main_parse['Sparql'],
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
