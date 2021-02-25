#%%

import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s %(message)s')

# Print to files
file_handler = logging.FileHandler('problem_tester.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
ch.setFormatter(formatter)
logger.addHandler(ch)
# # Add rich handler
# logger.addHandler(RichHandler())

logger.info("PUSH Testing ... http://google.com")
logger.error("IMPORTANT!")
# %%
