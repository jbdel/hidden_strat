import os
import unicodedata
import re
import urllib.request
from tqdm import tqdm


def download_ressource(checkpoint, url):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    os.makedirs(os.path.dirname(checkpoint))
    print('Downloading', url)
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, checkpoint, reporthook=t.update_to)


def slugify(value):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    https://github.com/django/django/blob/master/django/utils/text.py#L394
    """
    value = str(value)
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def get_output_dir(args):
    return os.path.join(args.model_dir, args.model, args.name if args.name is not None else '')


def get_report(report, policy=None):
    if policy is None:
        policy = 'top_section'

    if policy == 'top_section':
        for section in ['findings', 'impression', 'background']:
            if section not in report:
                continue

            if report[section] != '':
                return report[section]

    elif policy == 'all_section':
        ret = ''
        for section in ['findings', 'impression', 'background']:
            if section not in report:
                continue
            ret += report[section] + ' '
    else:
        raise NotImplementedError(policy)
