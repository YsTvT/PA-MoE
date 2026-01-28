import random
import re
import time
import os
import string

def get_time_str():
    tt = time.localtime()
    time_str = ('%04d_%02d_%02d_%02d_%02d_%02d' %
                (tt.tm_year, tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
    return time_str

def encode(string, encoding='utf-8'):
    return string.encode(encoding)

def decode(string, encoding='utf-8'):
    return string.decode(encoding)

def multireplace(string, replacements):
    
    substrs = sorted(replacements, key=len, reverse=True)

    regexp = re.compile('|'.join(map(re.escape, substrs)))

    return regexp.sub(lambda match: replacements[match.group(0)], string)

class SetWithGet(set):
    def get_any(self):
        return random.sample(self, 1)[0]

    def __getitem__(self, item):
        return self.get_any()

class Noop(object):
    def noop(*args, **kw):
        pass

    def __getattr__(self, _):
        return self.noop

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def remove_spaces(s):
    cs = ' '.join(s.split())
    return cs

def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs

def remove_punctuation(s):
    cs = s.translate(str.maketrans('', '', string.punctuation))
    cs = remove_spaces_and_lower(cs)
    return cs