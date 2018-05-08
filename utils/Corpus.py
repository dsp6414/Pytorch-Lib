import sys
import argparse
import os
import re
import string
import glob

if sys.version_info[0] >= 3:
    unicode = str

STOPWORDS = frozenset([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fify', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once'
])


RE_PUNCT = re.compile(r'([%s])+' % re.escape(string.punctuation), re.UNICODE)
RE_TAGS = re.compile(r"<([^>]+)>", re.UNICODE)
RE_NUMERIC = re.compile(r"[0-9]+", re.UNICODE)
RE_NONALPHA = re.compile(r"\W", re.UNICODE)
RE_AL_NUM = re.compile(r"([a-z]+)([0-9]+)", flags=re.UNICODE)
RE_NUM_AL = re.compile(r"([0-9]+)([a-z]+)", flags=re.UNICODE)
RE_WHITESPACE = re.compile(r"(\s)+", re.UNICODE)

def to_unicode(text, encoding='utf8', errors='strict'):

    if isinstance(text, unicode):
        return text
    return unicode(text, encoding, errors=errors)

def simple_clean(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def remove_stopwords(s):
    s = to_unicode(s)
    return " ".join(w for w in s.split() if w not in STOPWORDS)

def strip_punctuation(s):
    s = to_unicode(s)
    return RE_PUNCT.sub(" ", s)

def strip_tags(s):
    s = to_unicode(s)
    return RE_TAGS.sub("", s)

def remove_short(s, minsize=3):
    s = to_unicode(s)
    return " ".join(e for e in s.split() if len(e) >= minsize)

def strip_numeric(s):
    s = to_unicode(s)
    return RE_NUMERIC.sub("", s)

def strip_non_alphanum(s):
    s = to_unicode(s)
    return RE_NONALPHA.sub(" ", s)

def strip_multiple_whitespaces(s):
    s = to_unicode(s)
    return RE_WHITESPACE.sub(" ", s)

def split_alphanum(s):
    """
    >>> split_alphanum("24.0hours7 days365 a1b2c3")
    u'24.0 hours 7 days 365 a 1 b 2 c 3'

    """
    s = to_unicode(s)
    s = RE_AL_NUM.sub(r"\1 \2", s)
    return RE_NUM_AL.sub(r"\1 \2", s)

DEFAULT_FILTERS = [
    lambda x: x.lower(), strip_punctuation,
    strip_multiple_whitespaces, strip_numeric, remove_short]

class Corpus(object):

    def __init__(self, file_name,save_data, max_len=16):
        self.file_name = file_name
        self._save_data = save_data
        self._sents=[]
        self._labels=[]
        self.max_len = max_len
        
           

    def parse_data_from_file(self):
       
        _sents, _labels = [], []
        for sentence in open(self.file_name):
            label, _, _words = sentence.replace('\xf0', ' ').partition(' ') #特定格式：类别 文档，可改写该段代码
            label = label.split(":")[0]

            words = self.preprocess_string(_words) #处理文本

            if len(words) > self.max_len:
                words = words[:self.max_len]

            _sents += [words]
            _labels += [[label]]
        
        self._sents.extend(_sents)
        self._labels.extend(_labels)      

    def parse_data_from_dir(self,dirs,lines_are_documents=True):
        _sents, _labels = [], []
        dirs = os.path.expanduser(dirs)
        for label in sorted(os.listdir(dirs)):
            d = os.path.join(dirs, label)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    with open(path, 'rt') as f:
                        if lines_are_documents:
                            for line in f:
                                _sents += self.preprocess_string(line) 
                                _labels += [label]                              
                        else:
                            _sents +=[self.preprocess_string(f.read())] 
                            _labels += [[label]]
        
        self._sents.extend(_sents)
        self._labels.extend(_labels) 
              

    def save(self):

        data = {
            'max_len': self.max_len,
            'docs': self._sents,
            'label': self._labels
        }
        torch.save(data, self._save_data)


    def preprocess_string(self,s,filters=DEFAULT_FILTERS): #可以为None
        """
        Examples
        --------
        >>> preprocess_string("<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?")
        [u'hel', u'rld', u'weather', u'todai', u'isn']
        >>>
        >>> s = "<i>Hel 9lo</i> <b>Wo9 rld</b>! Th3     weather_is really g00d today, isn't it?"
        >>> CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation]
        >>> preprocess_string(s, CUSTOM_FILTERS)
        [u'hel', u'9lo', u'wo9', u'rld', u'th3', u'weather', u'is', u'really', u'g00d', u'today', u'isn', u't', u'it']

        """
        s = to_unicode(s)
        if filters is not None:
            for f in filters:
                s = f(s)
        return s.split()


  

