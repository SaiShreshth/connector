"""
Minimal generated-like protobuf classes for intent_message.proto used in tests.
This is a small, checked-in helper that simulates protobuf objects with
SerializeToString() for testing the producer. In a real setup you'd generate
these with protoc.
"""
import json

class RuleBasedResult:
    def __init__(self):
        self.action_code = ''
        self.confidence = 0.0
        self.matched_keywords = []

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.action_code = d.get('action_code', '')
        obj.confidence = float(d.get('confidence', 0.0))
        obj.matched_keywords = list(d.get('matched_keywords', []))
        return obj


class QueueMessage:
    def __init__(self):
        self.request_id = ''
        self.user_query = ''
        self.session_id = ''
        self.user_id = ''
        self.conversation_history = []
        self.rule_based_result = None
        self.timestamp = ''
        self.priority = ''
        self.metadata = {}

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.request_id = d.get('request_id', '')
        obj.user_query = d.get('user_query', '')
        obj.session_id = d.get('session_id', '') or ''
        obj.user_id = d.get('user_id', '') or ''
        obj.conversation_history = d.get('conversation_history', []) or []
        rb = d.get('rule_based_result') or {}
        obj.rule_based_result = RuleBasedResult.from_dict(rb) if rb else None
        obj.timestamp = d.get('timestamp', '')
        obj.priority = d.get('priority', '')
        obj.metadata = d.get('metadata', {}) or {}
        return obj

    def SerializeToString(self):
        # For testing we serialize to json bytes to emulate protobuf bytes
        d = {
            'request_id': self.request_id,
            'user_query': self.user_query,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'conversation_history': self.conversation_history,
            'rule_based_result': {
                'action_code': self.rule_based_result.action_code if self.rule_based_result else '',
                'confidence': self.rule_based_result.confidence if self.rule_based_result else 0.0,
                'matched_keywords': self.rule_based_result.matched_keywords if self.rule_based_result else []
            },
            'timestamp': self.timestamp,
            'priority': self.priority,
            'metadata': self.metadata
        }
        return json.dumps(d).encode('utf-8')


class BatchPayload:
    def __init__(self):
        self.batch_id = ''
        self.timestamp = ''
        self.messages = []

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.batch_id = d.get('batch_id', '')
        obj.timestamp = d.get('timestamp', '')
        obj.messages = [QueueMessage.from_dict(m) for m in d.get('messages', [])]
        return obj

    def SerializeToString(self):
        d = {
            'batch_id': self.batch_id,
            'timestamp': self.timestamp,
            'messages': [json.loads(m.SerializeToString().decode('utf-8')) for m in self.messages]
        }
        return json.dumps(d).encode('utf-8')
