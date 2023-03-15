import pdb 
import sys

seed_dict = {'los-angeles-1':8088, 'los-angeles-2':5100, 'gulfport':7975, 
    'cat-island':3753, 'pavia':9928, 'texas-goast':4448,}

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def get_params(net):
    '''Returns parameters that we want to optimize over.
    '''
    params = []
    params += [x for x in net.parameters()]
            
    return params

def img2mask(img):
    img = img[0].sum(0)
    img = img - img.min()
    img = img / img.max()
    img = img.detach().cpu().numpy()

    return img