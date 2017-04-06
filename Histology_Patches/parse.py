import sys
import re

fn = sys.argv[1]
C = open(fn).read()
results = re.findall(r'src="([^"]+)"', C)
results = [r.replace('https://atlases.muni.cz', '').replace('&amp;', '&') for r in results]
results = ['curl "https://atlases.muni.cz%s" >%s.jpg' % (r, idx) for idx, r in enumerate(results)]
print('\n'.join(results))