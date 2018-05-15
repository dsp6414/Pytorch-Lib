import sys
import argparse
import os
import re
import string
import glob

if sys.version_info[0] >= 3:
    unicode = str


if sys.version_info >= (3,3):
    dashes = ["–", "--+"]
    for i in range(8208, 8214):
        dashes.append(chr(i))
else:
    dashes = [u"–", u"--+"]
    for i in range(8208, 8214):
        dashes.append(unichr(i))


UNDECIDED = 0
SHOULD_SPLIT = 1
SHOULD_NOT_SPLIT = 2

people = [
    "jr", "mr", "ms", "mrs", "dr", "prof", "esq", "sr",
    "sen", "sens", "rep", "reps", "gov", "attys", "attys",
    "supt", "det", "mssrs", "rev", "fr", "ss", "msgr"
]
army   = ["col", "gen", "lt", "cmdr", "adm", "capt", "sgt", "cpl", "maj", "brig", "pt"]
inst   = ["dept","univ", "assn", "bros", "ph.d"]
place  = [
    "arc", "al", "ave", "blvd", "bld", "cl", "ct",
    "cres", "exp", "expy", "dist", "mt", "mtn", "ft",
    "fy", "fwy", "hwy", "hway", "la", "pde", "pd","plz", "pl", "rd", "st",
    "tce"
]
comp   = ["mfg", "inc", "ltd", "co", "corp"]
state  = [
    "ala","ariz","ark","cal","calif","colo","col","conn",
    "del","fed","fla","ga","ida","id","ill","ind","ia","kans",
    "kan","ken","ky","la","me","md","is","mass","mich","minn",
    "miss","mo","mont","neb","nebr","nev","mex","okla","ok",
    "ore","penna","penn","pa","dak","tenn","tex","ut","vt",
    "va","wash","wis","wisc","wy","wyo","usafa","alta",
    "man","ont","que","sask","yuk"
]
month  = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec"
]
misc = ["vs", "etc", "no","esp", "ed", "iv", "Oper", "op", "i.e", "e.g", "v"]
website = ["www"]
currency = ["rs"]
ABBR = {}
# create a hash of these abbreviations:
for abbreviation_type in [people, army, inst, place, comp, state, month, misc, website, currency]:
    for abbreviation in abbreviation_type:
        ABBR[abbreviation] = True

MONTHS = {
    "january", "february", "march", "april", "may",
    "june", "july", "august", "september", "october",
    "november", "december"
}
PUNCT_SYMBOLS = {'.', "...", "?", "!", "..", "!!", "??", "!?", "?!", u"…"}
CONTINUE_PUNCT_SYMBOLS = {';', ',', '-', ':'} | set(dashes)
OPENING_SYMBOLS = {'(', '[', '"', '{', '“'}
CLOSING_SYMBOLS = {')', ']', '"', '}', '”'}
CLOSE_2_OPEN = {')':'(', ']': '[', '"':'"', '}':'{', '”':'“'}


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


dashes_no_repeats = dashes[:]
dashes_no_repeats.remove("--+")

matching_dashes = dashes_no_repeats + ["-+"]

word_with_alpha_and_period   = re.compile("^([^\.]+)(\.\s*)$")
one_letter_long_or_repeating = re.compile("^(?:(?:[a-z])|(?:[a-z](?:\.[a-z])+))$", re.IGNORECASE)
no_punctuation               = re.compile("^\w+$")
left_quote_shifter           = re.compile(u"((`‘(?!`))|(‘(?!‘))\s*)(?=.*\w)", re.UNICODE)
left_quote_converter         = re.compile(u'([«"“]\s*)(?=.*\w)', re.UNICODE)
left_single_quote_converter  = re.compile(u"(?:(\W|^))('\s*)(?=.*\w)", re.UNICODE)
right_single_quote_converter = re.compile(u"(['’]+)(?=\W|$)\s*", re.UNICODE)

if sys.version_info >= (3,3):
    repeated_dash_converter = re.compile("--+")
    dash_converter = re.compile("|".join(dashes_no_repeats))
else:
    repeated_dash_converter = re.compile(u"--+")
    dash_converter = re.compile(u"|".join(dashes_no_repeats))

simple_dash_finder           = re.compile("(-\s*)")
advanced_dash_finder         = re.compile("(" + "|".join(matching_dashes) + ")\s*")
multi_single_quote_finder    = re.compile("('{2,})\s*")
url_file_finder              = re.compile("(?:[-a-zA-Z0-9@%._\+~#=]{2,256}://)?"
                                          "(?:www\.)?[-a-zA-Z0-9@:%\._\+~#=]{2,"
                                          "256}\.[a-z]{2,6}[-a-zA-Z0-9@:%_\+.~#"
                                          "?&//=]*\s*")
numerical_expression         = re.compile(u"(\d+(?:,\d+)*(?:\.\d+)*(?![a-zA-ZÀ-ż])\s*)")
remaining_quote_converter    = re.compile(u'(.)(?=["“”»])')
shifted_ellipses             = re.compile("([\.\!\?¿¡]{2,})\s*")
shifted_standard_punctuation = re.compile(u"([\(\[\{\}\]\)\!¡\?¿#\$%;~&+=<>|/:,—…])\s*")
period_mover                 = re.compile(u"([a-zA-ZÀ-ż]{2})([\./])\s+([a-zA-ZÀ-ż]{2})")
pure_whitespace              = re.compile("\s+")
english_specific_appendages = re.compile(u"(\w)(?=['’]([dms])\\b)", re.UNICODE)
english_nots = re.compile(u"(.)(?=n['’]t\\b)", re.UNICODE)
english_contractions = re.compile(u"(.)(?=['’](ve|ll|re)\\b)")
french_appendages = re.compile(u"(\\b[tjnlsmdclTJNLSMLDC]|qu)['’](?=[^tdms])")
word_with_period = re.compile("[^\s\.]+\.{0,1}")

def strip_word_with_alpha_and_period(s):
    s = to_unicode(s)
    return word_with_alpha_and_period.sub(" ", s)

def strip_one_letter_long_or_repeating(s):
    s = to_unicode(s)
    return one_letter_long_or_repeating.sub(" ", s)

def strip_no_punctuation(s):
    s = to_unicode(s)
    return no_punctuation.sub(" ", s)

def strip_left_quote_shifter(s):
    s = to_unicode(s)
    return left_quote_shifter.sub(" ", s)

def strip_left_quote_converter(s):
    s = to_unicode(s)
    return left_quote_converter.sub(" ", s)

def strip_left_single_quote_converter(s):
    s = to_unicode(s)
    return left_single_quote_converter.sub(" ", s)

def strip_right_single_quote_converter(s):
    s = to_unicode(s)
    return right_single_quote_converter.sub(" ", s)

def strip_repeated_dash_converter(s):
    s = to_unicode(s)
    return repeated_dash_converter.sub(" ", s)

def strip_dash_converter(s):
    s = to_unicode(s)
    return dash_converter.sub(" ", s)

def strip_simple_dash_finder(s):
    s = to_unicode(s)
    return simple_dash_finder.sub(" ", s)

def strip_advanced_dash_finder(s):
    s = to_unicode(s)
    return advanced_dash_finder.sub(" ", s)

def strip_multi_single_quote_finder(s):
    s = to_unicode(s)
    return multi_single_quote_finder.sub(" ", s)

def strip_shifted_ellipses(s):
    s = to_unicode(s)
    return shifted_ellipses.sub(" ", s)

def strip_shifted_standard_punctuation(s):
    s = to_unicode(s)
    return shifted_standard_punctuation.sub(" ", s)

def strip_period_mover(s):
    s = to_unicode(s)
    return period_mover.sub(" ", s)

def strip_pure_whitespace(s):
    s = to_unicode(s)
    return pure_whitespace.sub(" ", s)

def strip_english_specific_appendages(s):
    s = to_unicode(s)
    return english_specific_appendages.sub(" ", s)

def strip_english_nots(s):
    s = to_unicode(s)
    return english_nots.sub(" ", s)

def strip_english_contractions(s):
    s = to_unicode(s)
    return english_contractions.sub(" ", s)

def strip_word_with_period(s):
    s = to_unicode(s)
    return word_with_period.sub(" ", s)


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

if __name__ == '__main__':
   

    print('finished')


  

