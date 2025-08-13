import os
import sys

sys.path.append('/workspaces/trade-agent')
os.chdir('/workspaces/trade-agent')
exec(open('scripts/analyze_study_performance.py').read())
