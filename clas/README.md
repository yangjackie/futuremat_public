# Chemical Looping Ammonia Synthesis (CLAS)
## Code explain: 
### get_structure.py:
This is a command line tool to help download materials information from Materials Project
using its API.

### construct_cl.py:
Search and pair materials for CLs. It would query the ComputedStructureEntry
from MP and construct sub-reactions based on the stoichiometries.


## Code configuration:
Enter the api_key on the top of both python scripts: get_structure.py and construct_cl.py.